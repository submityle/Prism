//! Classic scanline polygon rasterization (Edge Table + Active Edge Table) for
//! the particle 2D authoring / footprint contracts (design §8.2, §12-§13).
//!
//! Several particle stages describe a shape as an ordered ring of 2D vertices
//! and then need the *interior pixels* of that ring on a `CPU` reference grid:
//! a footprint-baking pass rasterizes an emitter outline into a coverage mask;
//! a debug overlay wants the solid fill of an authored region; a deterministic
//! stencil reference wants the exact per-row fill intervals the `GPU` kernel
//! must reproduce bit for bit. This module owns the small, `CPU`-verifiable
//! contract those stages share: turning a simple polygon ring into the ordered
//! list of horizontal fill spans that cover its interior.
//!
//! The algorithm is the textbook scanline fill. Every non-horizontal ring edge
//! is entered into an *Edge Table* keyed by its lower `y`. Scanlines are
//! sampled at pixel centers `y + 0.5`; as the sweep advances, edges are moved
//! into an *Active Edge Table* when their span opens and dropped when it closes,
//! using the half-open interval `[y_lower, y_upper)` so a shared vertex is
//! never double-counted. On each scanline the active edges' `x` crossings are
//! sorted and paired under the even-odd (parity) rule, and each pair becomes one
//! [`Span`] of interior pixels. Concave rings therefore yield several disjoint
//! spans on a single row, exactly as required.
//!
//! # Pixel convention
//! A pixel column `x` is filled when its center `x + 0.5` lies inside the
//! crossing interval `[x_left, x_right]`; a row `y` is sampled at its center
//! `y + 0.5`. Coordinates are turned into integer pixel indices with the
//! round-half rule `(v + 0.5).floor()` (and the symmetric `(v - 0.5).floor()`
//! for the right edge), never with `f32` equality. With integer-vertex inputs
//! the filled pixel count matches the enclosed area up to the usual boundary
//! rounding.
//!
//! # Vertex extrema and horizontal edges
//! Horizontal edges carry no `x`-per-`y` information and are skipped entirely.
//! A local-maximum vertex (both incident edges descend from it) is the
//! `y_upper` of both, so the half-open rule excludes it and it contributes no
//! spurious crossing; a local-minimum vertex is the `y_lower` of both and is
//! counted for both, so a mere touch produces a zero-width (dropped) span rather
//! than a leak; a monotone pass-through vertex is counted exactly once. This is
//! the classic once-or-twice extremum rule.
//!
//! # Strict scope
//! This module *fills a simple polygon ring* into pixel spans. It is distinct
//! from, and neither imports nor reconstructs, its siblings: `point_in_polygon`
//! answers containment for a single query point, `polygon_area_2d` measures a
//! ring's scalar metrics, and `midpoint_circle` rasterizes a *circular disk*,
//! not an arbitrary polygon. Its private helpers are not shared outward.
//!
//! # No transcendental math
//! Every step is `+`, `-`, `*`, one `/` per edge for the inverse slope, and
//! `f32::floor` / `min` / `max` for pixel snapping. There is no `sin`, `cos`,
//! `atan`, `exp`, `ln`, `powf`, `sqrt` or any other transcendental call, and no
//! `f32` equality: near-zero magnitudes are compared against [`CMP_EPS`].

use alloc::vec::Vec;

/// Magnitude below which a `y`-extent, a slope denominator, or a scanline /
/// vertex `y` difference is treated as zero.
///
/// This is the comparison rule used throughout instead of `==` on `f32`: a
/// scalar is "zero" when its absolute value does not exceed this bound, which is
/// how horizontal edges and half-open interval boundaries are detected without
/// ever comparing two floats for exact equality.
pub const CMP_EPS: f32 = 1.0e-6;

/// A hand-rolled 2D vector, local to this module so it never depends on a
/// sibling's math type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    /// The `x` component.
    pub x: f32,
    /// The `y` component.
    pub y: f32,
}

impl Vec2 {
    /// Builds a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// One horizontal run of filled interior pixels on a single scanline.
///
/// The run covers the inclusive pixel columns `x_start ..= x_end` on row `y`.
/// A [`Span`] is only ever emitted when it is non-empty, so `x_start <= x_end`
/// always holds and [`Span::width`] is at least `1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// The integer scanline (pixel row) this run lies on.
    pub y: i32,
    /// The leftmost filled pixel column, inclusive.
    pub x_start: i32,
    /// The rightmost filled pixel column, inclusive.
    pub x_end: i32,
}

impl Span {
    /// Number of pixels covered by this run, always `>= 1`.
    #[must_use]
    pub fn width(self) -> i32 {
        self.x_end - self.x_start + 1
    }
}

/// A non-horizontal ring edge, normalized so `y_lower <= y_upper`.
///
/// `x_lower` is the `x` coordinate at `y_lower` and `inv_slope` is `dx/dy`, so
/// the crossing at a scanline center `yc` in `[y_lower, y_upper)` is
/// `x_lower + (yc - y_lower) * inv_slope`.
#[derive(Clone, Copy, Debug)]
struct Edge {
    y_lower: f32,
    y_upper: f32,
    x_lower: f32,
    inv_slope: f32,
}

/// Builds the Edge Table: every non-horizontal ring edge, normalized so the
/// lower endpoint comes first, sorted ascending by `y_lower` for incremental
/// activation.
fn build_edges(polygon: &[Vec2]) -> Vec<Edge> {
    let n = polygon.len();
    let mut edges = Vec::new();
    if n < 3 {
        return edges;
    }
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        let dy = b.y - a.y;
        // Horizontal edges contribute no x-per-y crossing and are skipped.
        if dy.abs() <= CMP_EPS {
            continue;
        }
        let (lower, upper) = if a.y < b.y { (a, b) } else { (b, a) };
        let inv_slope = (upper.x - lower.x) / (upper.y - lower.y);
        edges.push(Edge {
            y_lower: lower.y,
            y_upper: upper.y,
            x_lower: lower.x,
            inv_slope,
        });
    }
    edges.sort_by(|p, q| p.y_lower.total_cmp(&q.y_lower));
    edges
}

/// Rasterizes a simple polygon ring into its interior fill spans.
///
/// The ring is passed as an *open* slice of vertices; the closing edge from the
/// last vertex back to the first is implied. The returned spans are
/// deterministically ordered by scanline `y` ascending and, within a scanline,
/// by `x_start` ascending, so a shape and any permutation of the *same* ring
/// winding (or its reverse) produce byte-identical output. Fewer than three
/// vertices, or a fully degenerate (collinear / zero-height) ring, enclose no
/// pixels and yield an empty vector.
#[must_use]
pub fn scanline_fill(polygon: &[Vec2]) -> Vec<Span> {
    let edges = build_edges(polygon);
    let mut spans = Vec::new();
    if edges.is_empty() {
        return spans;
    }

    // Vertical extent of the ring; scanline centers outside it fill nothing.
    let mut y_min = f32::INFINITY;
    let mut y_max = -f32::INFINITY;
    for v in polygon {
        y_min = y_min.min(v.y);
        y_max = y_max.max(v.y);
    }

    // A padded integer scanline range: rows whose center falls outside the
    // active edges simply produce no crossings and are dropped.
    let y_start = (y_min - 1.0).floor() as i32;
    let y_end = (y_max + 1.0).floor() as i32;

    let mut active: Vec<usize> = Vec::new();
    let mut next = 0usize;
    let mut xs: Vec<f32> = Vec::new();

    for y in y_start..=y_end {
        let yc = y as f32 + 0.5;

        // Move newly-opened edges from the Edge Table into the Active Edge
        // Table: an edge opens once its lower endpoint is at or below yc.
        while next < edges.len() && edges[next].y_lower <= yc + CMP_EPS {
            active.push(next);
            next += 1;
        }

        // Retire edges whose upper endpoint is at or below yc (half-open top,
        // so a shared local-maximum vertex is excluded exactly once).
        active.retain(|&ei| edges[ei].y_upper - yc > CMP_EPS);

        if active.is_empty() {
            continue;
        }

        // Gather this scanline's x crossings and sort them left to right.
        xs.clear();
        for &ei in &active {
            let e = &edges[ei];
            xs.push(e.x_lower + (yc - e.y_lower) * e.inv_slope);
        }
        xs.sort_by(f32::total_cmp);

        // Even-odd parity: interior lies between consecutive crossing pairs.
        let mut i = 0;
        while i + 1 < xs.len() {
            let x_left = xs[i];
            let x_right = xs[i + 1];
            // A pixel column is filled when its center lies inside the run.
            let x_start = (x_left + 0.5).floor() as i32;
            let x_end = (x_right - 0.5).floor() as i32;
            if x_start <= x_end {
                spans.push(Span { y, x_start, x_end });
            }
            i += 2;
        }
    }

    spans
}

/// Total number of interior pixels covered by [`scanline_fill`], i.e. the sum of
/// every span's [`Span::width`].
///
/// For an integer-vertex ring this approximates the enclosed area up to boundary
/// rounding, and it is exact for an axis-aligned integer rectangle.
#[must_use]
pub fn filled_pixel_count(polygon: &[Vec2]) -> usize {
    let mut total = 0usize;
    for s in scanline_fill(polygon) {
        total += s.width() as usize;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn ring(pts: &[(f32, f32)]) -> Vec<Vec2> {
        pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect()
    }

    const RECT_10: [(f32, f32); 4] = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
    const TRI: [(f32, f32); 3] = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)];
    const U_SHAPE: [(f32, f32); 8] = [
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 4.0),
        (3.0, 4.0),
        (3.0, 1.0),
        (1.0, 1.0),
        (1.0, 4.0),
        (0.0, 4.0),
    ];
    const DIAMOND: [(f32, f32); 4] = [(0.0, -4.0), (4.0, 0.0), (0.0, 4.0), (-4.0, 0.0)];

    fn spans_on(spans: &[Span], y: i32) -> Vec<Span> {
        spans.iter().copied().filter(|s| s.y == y).collect()
    }

    #[test]
    fn empty_polygon_is_empty() {
        assert!(scanline_fill(&[]).is_empty());
    }

    #[test]
    fn single_vertex_is_empty() {
        assert!(scanline_fill(&ring(&[(1.0, 1.0)])).is_empty());
    }

    #[test]
    fn two_vertices_is_empty() {
        assert!(scanline_fill(&ring(&[(0.0, 0.0), (5.0, 5.0)])).is_empty());
    }

    #[test]
    fn degenerate_collinear_horizontal_is_empty() {
        // All vertices share a y: every edge is horizontal and skipped.
        let line = ring(&[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]);
        assert!(scanline_fill(&line).is_empty());
    }

    #[test]
    fn degenerate_collinear_vertical_is_empty() {
        // A zero-width ring: crossings coincide, so every run is empty.
        let line = ring(&[(2.0, 0.0), (2.0, 10.0), (2.0, 5.0)]);
        assert!(scanline_fill(&line).is_empty());
    }

    #[test]
    fn rectangle_span_count_equals_height() {
        let spans = scanline_fill(&ring(&RECT_10));
        assert_eq!(spans.len(), 10);
    }

    #[test]
    fn rectangle_rows_span_zero_to_nine() {
        let spans = scanline_fill(&ring(&RECT_10));
        for (i, s) in spans.iter().enumerate() {
            assert_eq!(s.y, i as i32);
        }
    }

    #[test]
    fn rectangle_each_span_is_full_width() {
        let spans = scanline_fill(&ring(&RECT_10));
        for s in &spans {
            assert_eq!(s.x_start, 0);
            assert_eq!(s.x_end, 9);
            assert_eq!(s.width(), 10);
        }
    }

    #[test]
    fn rectangle_pixel_count_equals_area() {
        assert_eq!(filled_pixel_count(&ring(&RECT_10)), 100);
    }

    #[test]
    fn triangle_small_pixel_count_is_exact() {
        // Right triangle, pixel centers with i + j <= 9: sum_{s=0}^{9}(s+1)=55.
        assert_eq!(filled_pixel_count(&ring(&TRI)), 55);
    }

    #[test]
    fn triangle_rows_are_monotonically_narrowing() {
        let spans = scanline_fill(&ring(&TRI));
        // One span per row for a convex ring; widths strictly decrease.
        for pair in spans.windows(2) {
            assert!(pair[1].y > pair[0].y);
            assert!(pair[1].width() < pair[0].width());
        }
        assert_eq!(spans.first().unwrap().width(), 10);
        assert_eq!(spans.last().unwrap().width(), 1);
    }

    #[test]
    fn triangle_left_edge_anchored_at_zero() {
        let spans = scanline_fill(&ring(&TRI));
        for s in &spans {
            assert_eq!(s.x_start, 0);
        }
    }

    #[test]
    fn output_is_sorted_by_y_then_x() {
        let spans = scanline_fill(&ring(&U_SHAPE));
        for pair in spans.windows(2) {
            let before = (pair[0].y, pair[0].x_start);
            let after = (pair[1].y, pair[1].x_start);
            assert!(before <= after);
        }
    }

    #[test]
    fn concave_u_splits_notch_row_into_two_spans() {
        let spans = scanline_fill(&ring(&U_SHAPE));
        let row = spans_on(&spans, 2);
        assert_eq!(row.len(), 2);
        assert_eq!(row[0].x_start, 0);
        assert_eq!(row[0].x_end, 0);
        assert_eq!(row[1].x_start, 3);
        assert_eq!(row[1].x_end, 3);
    }

    #[test]
    fn concave_u_base_row_is_single_span() {
        let spans = scanline_fill(&ring(&U_SHAPE));
        let row = spans_on(&spans, 0);
        assert_eq!(row.len(), 1);
        assert_eq!(row[0].x_start, 0);
        assert_eq!(row[0].x_end, 3);
    }

    #[test]
    fn concave_u_total_span_count() {
        // Row 0: 1 span (base bar); rows 1..=3: 2 spans each -> 1 + 2*3 = 7.
        let spans = scanline_fill(&ring(&U_SHAPE));
        assert_eq!(spans.len(), 7);
    }

    #[test]
    fn concave_l_shape_stays_single_span_per_row() {
        // An L is concave but its rows are each a single interval.
        let l = ring(&[
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 2.0),
            (2.0, 2.0),
            (2.0, 4.0),
            (0.0, 4.0),
        ]);
        let spans = scanline_fill(&l);
        for y in 0..4 {
            assert_eq!(spans_on(&spans, y).len(), 1, "row {y} should be one span");
        }
        // Lower rows span the wide base; upper rows only the narrow column.
        assert_eq!(spans_on(&spans, 0)[0].width(), 4);
        assert_eq!(spans_on(&spans, 3)[0].width(), 2);
    }

    #[test]
    fn local_maximum_vertex_does_not_overfill() {
        // The diamond apex (0, 4) is a local maximum; the top row must be the
        // single center pixel, never a leak above y = 3.
        let spans = scanline_fill(&ring(&DIAMOND));
        assert!(spans.iter().all(|s| s.y <= 3));
        let top = spans_on(&spans, 3);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].width(), 1);
    }

    #[test]
    fn local_minimum_vertex_does_not_overfill() {
        // The diamond nadir (0, -4) is a local minimum; nothing below y = -4.
        let spans = scanline_fill(&ring(&DIAMOND));
        assert!(spans.iter().all(|s| s.y >= -4));
        let bottom = spans_on(&spans, -4);
        assert_eq!(bottom.len(), 1);
        assert_eq!(bottom[0].width(), 1);
    }

    #[test]
    fn horizontal_edges_and_collinear_vertices_are_ignored() {
        // An extra collinear vertex on the bottom edge and a split top edge must
        // not change the fill versus the plain rectangle.
        let split = ring(&[
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (5.0, 10.0),
            (0.0, 10.0),
        ]);
        assert_eq!(scanline_fill(&split), scanline_fill(&ring(&RECT_10)));
    }

    #[test]
    fn winding_direction_is_irrelevant() {
        let mut reversed = RECT_10.to_vec();
        reversed.reverse();
        assert_eq!(
            scanline_fill(&ring(&reversed)),
            scanline_fill(&ring(&RECT_10))
        );
    }

    #[test]
    fn output_is_deterministic_across_calls() {
        let poly = ring(&U_SHAPE);
        assert_eq!(scanline_fill(&poly), scanline_fill(&poly));
    }

    #[test]
    fn diamond_is_vertically_symmetric() {
        // Row y and its mirror row (-1 - y) must have equal total width.
        let spans = scanline_fill(&ring(&DIAMOND));
        let width_on = |y: i32| -> i32 { spans_on(&spans, y).iter().map(|s| s.width()).sum() };
        for y in -4..=3 {
            assert_eq!(width_on(y), width_on(-1 - y), "asymmetry at row {y}");
        }
    }

    #[test]
    fn centered_rectangle_is_horizontally_symmetric() {
        // Even-width rectangle centered on x = 0: each span mirrors under the
        // pixel reflection x -> -1 - x, i.e. x_start + x_end == -1.
        let rect = ring(&[(-4.0, -4.0), (4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)]);
        let spans = scanline_fill(&rect);
        assert!(!spans.is_empty());
        for s in &spans {
            assert_eq!(s.x_start + s.x_end, -1);
        }
    }

    #[test]
    fn translation_shifts_spans_uniformly() {
        let base = scanline_fill(&ring(&RECT_10));
        let moved_ring = ring(&[(5.0, 3.0), (15.0, 3.0), (15.0, 13.0), (5.0, 13.0)]);
        let moved = scanline_fill(&moved_ring);
        let shifted: Vec<Span> = base
            .iter()
            .map(|s| Span {
                y: s.y + 3,
                x_start: s.x_start + 5,
                x_end: s.x_end + 5,
            })
            .collect();
        assert_eq!(moved, shifted);
    }

    #[test]
    fn filled_pixel_count_matches_sum_of_widths() {
        let poly = ring(&U_SHAPE);
        let by_spans: usize = scanline_fill(&poly)
            .iter()
            .map(|s| s.width() as usize)
            .sum();
        assert_eq!(filled_pixel_count(&poly), by_spans);
    }

    #[test]
    fn large_rectangle_area_is_exact() {
        let rect = ring(&[(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]);
        assert_eq!(filled_pixel_count(&rect), 200 * 100);
    }

    #[test]
    fn large_triangle_count_approximates_area() {
        // Area 5000; center-sampled count 5050, well within a 10% band.
        let tri = ring(&[(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]);
        let count = filled_pixel_count(&tri) as f32;
        let area = 5000.0f32;
        assert!(
            (count - area).abs() <= 0.1 * area,
            "count {count} vs area {area}"
        );
    }

    #[test]
    fn span_width_helper_is_inclusive() {
        let s = Span {
            y: 0,
            x_start: 2,
            x_end: 7,
        };
        assert_eq!(s.width(), 6);
    }

    #[test]
    fn every_emitted_span_is_non_empty() {
        for poly in [ring(&RECT_10), ring(&TRI), ring(&U_SHAPE), ring(&DIAMOND)] {
            for s in scanline_fill(&poly) {
                assert!(s.x_start <= s.x_end);
                assert!(s.width() >= 1);
            }
        }
    }

    #[test]
    fn pass_through_vertex_counted_once() {
        // A rightward-pointing chevron: the tip (4, 1) is a monotone vertex on
        // the way up the right side, so each row is a single interval.
        let chevron = ring(&[(0.0, 0.0), (4.0, 1.0), (0.0, 2.0), (1.0, 1.0)]);
        let spans = scanline_fill(&chevron);
        assert!(!spans.is_empty());
        for y in spans.iter().map(|s| s.y).collect::<Vec<_>>() {
            assert_eq!(spans_on(&spans, y).len(), 1);
        }
    }

    #[test]
    fn parity_rule_handles_four_crossings() {
        // A plus / cross shape produces one span in its bar rows and, in the
        // wide middle band, still a single interval; verify the wide row.
        let plus = ring(&[
            (1.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (3.0, 1.0),
            (3.0, 2.0),
            (2.0, 2.0),
            (2.0, 3.0),
            (1.0, 3.0),
            (1.0, 2.0),
            (0.0, 2.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ]);
        let spans = scanline_fill(&plus);
        // Middle band (row 1) crosses the full width 0..=2.
        let mid = spans_on(&spans, 1);
        assert_eq!(mid.len(), 1);
        assert_eq!(mid[0].x_start, 0);
        assert_eq!(mid[0].x_end, 2);
        // A vertical-arm row (row 0) is just the central column.
        let arm = spans_on(&spans, 0);
        assert_eq!(arm.len(), 1);
        assert_eq!(arm[0].x_start, 1);
        assert_eq!(arm[0].x_end, 1);
    }
}
