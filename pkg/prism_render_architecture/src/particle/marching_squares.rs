//! 2D Marching Squares iso-contour extraction on a scalar sample grid (design
//! §8.2 authoring / field-visualization reference, `CPU`).
//!
//! Several particle stages hold a *2D scalar field* — a signed-distance
//! footprint, a density slice, a baked coverage or heat field — and need the
//! poly-line **iso-contour** at a threshold: the boundary curve where the field
//! equals `iso`. This module owns exactly that contract: given a row-major
//! `width × height` sample grid and a threshold, it walks every cell, classifies
//! its four corners against `iso` into a 4-bit case, and emits the linearly
//! interpolated contour [`Segment`]s in grid coordinates. It is the textbook
//! Marching Squares algorithm, disambiguating the two diagonal *saddle* cases
//! (`5` and `10`) with the cell-center average so the extracted curve stays
//! consistent with the sampled field.
//!
//! # Boundary with sibling modules
//! This is a strictly 2D, grid-in / segments-out contour extractor. It is *not*:
//! - [`super::volume_march`], which ray-marches a 3D volume front-to-back and
//!   composites opacity/transmittance — a stepping/compositing loop, not a
//!   geometric iso-surface;
//! - `marching_cubes` (the 3D sibling in this batch), which extracts a
//!   triangulated iso-*surface* from a voxel grid — this module is its 2D,
//!   segment-emitting analogue and shares no code with it;
//! - [`super::scanline_polygon_fill`], which rasterizes the *interior pixels* of
//!   an already-known polygon ring (Edge Table / Active Edge Table) rather than
//!   discovering a boundary from a field;
//! - [`super::conservative_raster`], which covers every pixel a triangle
//!   *touches* — a coverage predicate, not a contour.
//!
//! # No transcendental math
//! Every routine is pure `+`, `-`, `*`, `/`, comparison, `f32::abs`,
//! `f32::clamp`, and a single `f32::sqrt` (only in [`Point2::distance`], a
//! convenience the extractor itself never calls). There is no `sin`, `exp`,
//! `atan`, or `f32::floor`/`ceil` on the contour path: cell iteration is by
//! integer index and corner classification is a `>=` test, so no rounding is
//! required. Floats are never compared with `==`/`!=`; a corner is *inside* when
//! its value is `>= iso`, and an edge whose two corner values are within [`EPS`]
//! of each other is crossed at its midpoint rather than dividing by ~zero.

use alloc::vec::Vec;

/// Magnitude below which an edge's corner-value difference is treated as zero.
///
/// When the two corner values along an edge differ by less than this, the
/// linear crossing parameter is ill-conditioned (a near-`0/0`), so the crossing
/// is placed at the edge midpoint instead of dividing by a vanishing
/// denominator. Floats are otherwise never compared with `==`/`!=`.
pub const EPS: f32 = 1.0e-6;

/// A 2D point in grid coordinates, where integer coordinates land on grid
/// samples and a cell spans one unit in each axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2 {
    /// Horizontal (column) coordinate.
    pub x: f32,
    /// Vertical (row) coordinate.
    pub y: f32,
}

impl Point2 {
    /// Builds a point from its components.
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Component-wise sum `self + other`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub methods for call-site uniformity, matching the sibling vector types; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    /// Component-wise difference `self - other`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named sub for call-site uniformity, not the Sub operator trait."
    )]
    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    /// Linear interpolation `self + (other - self) * t`.
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        self.add(other.sub(self).scale(t))
    }

    /// Euclidean distance to `other` (the module's only `sqrt`).
    #[must_use]
    pub fn distance(self, other: Self) -> f32 {
        let d = other.sub(self);
        (d.x * d.x + d.y * d.y).sqrt()
    }
}

/// One contour line segment between two edge-crossing points, in grid
/// coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// First endpoint.
    pub a: Point2,
    /// Second endpoint.
    pub b: Point2,
}

impl Segment {
    /// Builds a segment from its two endpoints.
    #[must_use]
    pub fn new(a: Point2, b: Point2) -> Self {
        Self { a, b }
    }

    /// Euclidean length of the segment.
    #[must_use]
    pub fn length(self) -> f32 {
        self.a.distance(self.b)
    }
}

/// Classifies a cell's four corner values against `iso` into a 4-bit case.
///
/// Bit `0` is the bottom-left corner, bit `1` bottom-right, bit `2` top-right,
/// and bit `3` top-left; a bit is set when that corner is *inside*, i.e. its
/// value is `>= iso` (so a corner exactly equal to `iso` counts as inside). The
/// result is in `0..=15`.
#[must_use]
pub fn case_index(v0: f32, v1: f32, v2: f32, v3: f32, iso: f32) -> u8 {
    let mut c = 0u8;
    if v0 >= iso {
        c |= 1;
    }
    if v1 >= iso {
        c |= 2;
    }
    if v2 >= iso {
        c |= 4;
    }
    if v3 >= iso {
        c |= 8;
    }
    c
}

/// Linear crossing parameter along an edge from value `va` to value `vb`.
///
/// Returns the fraction `t` in `0.0..=1.0` at which the value equals `iso`,
/// computed as `(iso - va) / (vb - va)` and clamped to the edge. When the two
/// values differ by less than [`EPS`] the denominator is ill-conditioned, so
/// the crossing is placed at the midpoint `0.5`.
#[must_use]
pub fn lerp_param(va: f32, vb: f32, iso: f32) -> f32 {
    let denom = vb - va;
    if denom.abs() < EPS {
        0.5
    } else {
        ((iso - va) / denom).clamp(0.0, 1.0)
    }
}

/// Interpolated crossing point on cell edge `edge` (`0`=bottom, `1`=right,
/// `2`=top, `3`=left) for the cell whose bottom-left corner is `(cx, cy)`.
///
/// `v` holds the four corner values in the `[bottom-left, bottom-right,
/// top-right, top-left]` order used by [`case_index`]. The fall-through arm is
/// the left edge (`edge == 3`); callers only ever pass `0..=3`.
fn edge_point(edge: usize, cx: f32, cy: f32, v: [f32; 4], iso: f32) -> Point2 {
    match edge {
        0 => Point2::new(cx + lerp_param(v[0], v[1], iso), cy),
        1 => Point2::new(cx + 1.0, cy + lerp_param(v[1], v[2], iso)),
        2 => Point2::new(cx + 1.0 - lerp_param(v[2], v[3], iso), cy + 1.0),
        _ => Point2::new(cx, cy + 1.0 - lerp_param(v[3], v[0], iso)),
    }
}

/// Marching Squares edge-connection table.
///
/// Returns up to two edge pairs; each pair `[e0, e1]` becomes one contour
/// segment connecting the crossing on edge `e0` to the crossing on edge `e1`. A
/// pair of `[-1, -1]` marks "no segment". For the two diagonal saddle cases
/// (`5` and `10`) the connection depends on whether the cell *center* is inside,
/// which resolves the ambiguity toward the topology the sampled field implies.
#[must_use]
fn segment_edges(case: u8, center_inside: bool) -> [[i8; 2]; 2] {
    const NONE: [i8; 2] = [-1, -1];
    match case {
        1 | 14 => [[3, 0], NONE],
        2 | 13 => [[0, 1], NONE],
        4 | 11 => [[1, 2], NONE],
        7 | 8 => [[2, 3], NONE],
        3 | 12 => [[3, 1], NONE],
        6 | 9 => [[0, 2], NONE],
        5 => {
            if center_inside {
                [[0, 1], [2, 3]]
            } else {
                [[3, 0], [1, 2]]
            }
        }
        10 => {
            if center_inside {
                [[3, 0], [1, 2]]
            } else {
                [[0, 1], [2, 3]]
            }
        }
        _ => [NONE, NONE],
    }
}

/// Extracts the `iso` contour of a row-major `width × height` scalar field as a
/// list of line [`Segment`]s in grid coordinates.
///
/// Sample `(x, y)` is read from `field[y * width + x]` and sits at grid
/// coordinate `(x as f32, y as f32)`; each cell spans the unit square from its
/// bottom-left corner. Every cell is classified with [`case_index`], crossed
/// edges are interpolated with [`lerp_param`], and diagonal saddles are
/// disambiguated with the four-corner center average. A grid with fewer than two
/// samples on either axis, or a `field` slice shorter than `width * height`,
/// yields no segments.
#[must_use]
pub fn extract_contours(field: &[f32], width: usize, height: usize, iso: f32) -> Vec<Segment> {
    let mut out = Vec::new();
    if width < 2 || height < 2 {
        return out;
    }
    if field.len() < width * height {
        return out;
    }
    for cy in 0..(height - 1) {
        for cx in 0..(width - 1) {
            let v0 = field[cy * width + cx];
            let v1 = field[cy * width + (cx + 1)];
            let v2 = field[(cy + 1) * width + (cx + 1)];
            let v3 = field[(cy + 1) * width + cx];
            let case = case_index(v0, v1, v2, v3, iso);
            if case == 0 || case == 15 {
                continue;
            }
            let v = [v0, v1, v2, v3];
            let center = (v0 + v1 + v2 + v3) * 0.25;
            let center_inside = center >= iso;
            let fx = cx as f32;
            let fy = cy as f32;
            for pair in segment_edges(case, center_inside) {
                let e0 = pair[0];
                let e1 = pair[1];
                if e0 < 0 || e1 < 0 {
                    continue;
                }
                let a = edge_point(e0 as usize, fx, fy, v, iso);
                let b = edge_point(e1 as usize, fx, fy, v, iso);
                out.push(Segment::new(a, b));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const TEST_EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    fn point_approx(p: Point2, x: f32, y: f32) -> bool {
        approx(p.x, x) && approx(p.y, y)
    }

    fn points_equal(a: Point2, b: Point2) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y)
    }

    /// Builds a row-major 2x2 field from corner values named in the
    /// `[bottom-left, bottom-right, top-right, top-left]` order used by
    /// [`case_index`]. Row-major layout stores row 0 (`v0`, `v1`) then
    /// row 1 (`v3`, `v2`).
    fn cell2(v0: f32, v1: f32, v2: f32, v3: f32) -> Vec<f32> {
        vec![v0, v1, v3, v2]
    }

    /// True when the segment connects `p` and `q` in either orientation.
    fn segment_connects(s: Segment, p: Point2, q: Point2) -> bool {
        (points_equal(s.a, p) && points_equal(s.b, q))
            || (points_equal(s.a, q) && points_equal(s.b, p))
    }

    #[test]
    fn all_below_iso_yields_no_segments() {
        let field = vec![0.0_f32; 9];
        let segs = extract_contours(&field, 3, 3, 0.5);
        assert!(segs.is_empty());
    }

    #[test]
    fn all_above_iso_yields_no_segments() {
        let field = vec![10.0_f32; 9];
        let segs = extract_contours(&field, 3, 3, 0.5);
        assert!(segs.is_empty());
    }

    #[test]
    fn all_exactly_iso_are_inside_no_contour() {
        // Every corner value equals iso, so all are "inside" (>=), case 15.
        let field = vec![0.5_f32; 16];
        let segs = extract_contours(&field, 4, 4, 0.5);
        assert!(segs.is_empty());
    }

    #[test]
    fn single_bottom_left_corner_high_one_segment() {
        // v0 high, rest low -> case 1 -> segment left(3)->bottom(0).
        let field = vec![1.0, 0.0, 0.0, 0.0];
        let segs = extract_contours(&field, 2, 2, 0.5);
        assert_eq!(segs.len(), 1);
        // With v0=1, others 0, iso 0.5: bottom t = 0.5 -> (0.5, 0);
        // left t = (0.5-0)/(1-0)=0.5 from v3->v0 -> (0, 1-0.5)=(0,0.5).
        assert!(segment_connects(
            segs[0],
            Point2::new(0.0, 0.5),
            Point2::new(0.5, 0.0)
        ));
    }

    #[test]
    fn single_bottom_right_corner_high_case2() {
        // v1 high -> case 2 -> bottom(0)->right(1).
        let field = vec![0.0, 1.0, 0.0, 0.0];
        let segs = extract_contours(&field, 2, 2, 0.5);
        assert_eq!(segs.len(), 1);
        // bottom t = (0.5-0)/(1-0)=0.5 -> (0.5,0); right t=(0.5-1)/(0-1)=0.5 -> (1,0.5).
        assert!(segment_connects(
            segs[0],
            Point2::new(0.5, 0.0),
            Point2::new(1.0, 0.5)
        ));
    }

    #[test]
    fn single_top_right_corner_high_case4() {
        // v2 high -> case 4 -> right(1)->top(2).
        let field = cell2(0.0, 0.0, 1.0, 0.0);
        let segs = extract_contours(&field, 2, 2, 0.5);
        assert_eq!(segs.len(), 1);
        // right t=(0.5-0)/(1-0)=0.5 -> (1,0.5); top t=0.5 -> (1-0.5,1)=(0.5,1).
        assert!(segment_connects(
            segs[0],
            Point2::new(1.0, 0.5),
            Point2::new(0.5, 1.0)
        ));
    }

    #[test]
    fn single_top_left_corner_high_case8() {
        // v3 high -> case 8 -> top(2)->left(3).
        let field = cell2(0.0, 0.0, 0.0, 1.0);
        let segs = extract_contours(&field, 2, 2, 0.5);
        assert_eq!(segs.len(), 1);
        // top t=(0.5-0)/(1-0)=0.5 (v2->v3) -> (1-0.5,1)=(0.5,1);
        // left t from v3(1)->v0(0): (0.5-1)/(0-1)=0.5 -> (0, 1-0.5)=(0,0.5).
        assert!(segment_connects(
            segs[0],
            Point2::new(0.5, 1.0),
            Point2::new(0.0, 0.5)
        ));
    }

    #[test]
    fn bottom_edge_high_case3_horizontal_split() {
        // v0,v1 high (bottom row) -> case 3 -> left(3)->right(1).
        let field = vec![1.0, 1.0, 0.0, 0.0];
        let segs = extract_contours(&field, 2, 2, 0.5);
        assert_eq!(segs.len(), 1);
        // left crosses at y=0.5, right at y=0.5.
        assert!(segment_connects(
            segs[0],
            Point2::new(0.0, 0.5),
            Point2::new(1.0, 0.5)
        ));
    }

    #[test]
    fn right_column_high_case6_vertical_split() {
        // v1,v2 high (right column) -> case 6 -> bottom(0)->top(2).
        let field = cell2(0.0, 1.0, 1.0, 0.0);
        let segs = extract_contours(&field, 2, 2, 0.5);
        assert_eq!(segs.len(), 1);
        assert!(segment_connects(
            segs[0],
            Point2::new(0.5, 0.0),
            Point2::new(0.5, 1.0)
        ));
    }

    #[test]
    fn three_corners_high_isolates_missing_corner_case7() {
        // v0,v1,v2 high, v3 low -> case 7 -> top(2)->left(3), isolates top-left.
        let field = cell2(1.0, 1.0, 1.0, 0.0);
        let segs = extract_contours(&field, 2, 2, 0.5);
        assert_eq!(segs.len(), 1);
        // top from v2(1)->v3(0): t=(0.5-1)/(0-1)=0.5 -> (0.5,1);
        // left from v3(0)->v0(1): t=0.5 -> (0,0.5).
        assert!(segment_connects(
            segs[0],
            Point2::new(0.5, 1.0),
            Point2::new(0.0, 0.5)
        ));
    }

    #[test]
    fn case_index_matches_corner_membership() {
        assert_eq!(case_index(0.0, 0.0, 0.0, 0.0, 0.5), 0);
        assert_eq!(case_index(1.0, 0.0, 0.0, 0.0, 0.5), 1);
        assert_eq!(case_index(0.0, 1.0, 0.0, 0.0, 0.5), 2);
        assert_eq!(case_index(0.0, 0.0, 1.0, 0.0, 0.5), 4);
        assert_eq!(case_index(0.0, 0.0, 0.0, 1.0, 0.5), 8);
        assert_eq!(case_index(1.0, 1.0, 1.0, 1.0, 0.5), 15);
        assert_eq!(case_index(1.0, 0.0, 1.0, 0.0, 0.5), 5);
        assert_eq!(case_index(0.0, 1.0, 0.0, 1.0, 0.5), 10);
    }

    #[test]
    fn case_index_treats_exact_iso_as_inside() {
        // v0 exactly equals iso -> counted inside -> bit 0 set.
        assert_eq!(case_index(0.5, 0.0, 0.0, 0.0, 0.5), 1);
    }

    #[test]
    fn lerp_param_midpoint() {
        assert!(approx(lerp_param(0.0, 1.0, 0.5), 0.5));
    }

    #[test]
    fn lerp_param_quarter_and_three_quarter() {
        // iso 1 between 0 and 4 -> t = 0.25.
        assert!(approx(lerp_param(0.0, 4.0, 1.0), 0.25));
        // iso 3 between 0 and 4 -> t = 0.75.
        assert!(approx(lerp_param(0.0, 4.0, 3.0), 0.75));
    }

    #[test]
    fn lerp_param_clamps_out_of_range() {
        // iso below both -> clamps to 0; above both -> clamps to 1.
        assert!(approx(lerp_param(2.0, 5.0, 1.0), 0.0));
        assert!(approx(lerp_param(2.0, 5.0, 9.0), 1.0));
    }

    #[test]
    fn lerp_param_degenerate_equal_values_uses_midpoint() {
        // Denominator ~0 -> midpoint fallback, no division blowup.
        let t = lerp_param(0.3, 0.3, 0.3);
        assert!(approx(t, 0.5));
        assert!(t.is_finite());
    }

    #[test]
    fn interpolation_positions_are_exact_case1_asymmetric() {
        // v0=4, others 0, iso=1 -> asymmetric crossings.
        let field = vec![4.0, 0.0, 0.0, 0.0];
        let segs = extract_contours(&field, 2, 2, 1.0);
        assert_eq!(segs.len(), 1);
        // bottom from v0(4)->v1(0): t=(1-4)/(0-4)=0.75 -> (0.75,0).
        // left from v3(0)->v0(4): t=(1-0)/(4-0)=0.25 -> (0, 1-0.25)=(0,0.75).
        assert!(segment_connects(
            segs[0],
            Point2::new(0.75, 0.0),
            Point2::new(0.0, 0.75)
        ));
    }

    #[test]
    fn saddle_case5_center_low_isolates_each_inside_corner() {
        // v0,v2 high; v1,v3 low; center avg = 0.5 < iso 0.6 -> two separate corners.
        let field = cell2(1.0, 0.0, 1.0, 0.0);
        let segs = extract_contours(&field, 2, 2, 0.6);
        assert_eq!(segs.len(), 2);
        // center low -> segments left(3)->bottom(0) and right(1)->top(2).
        // The two segments are near the low corners' opposite arrangement.
        // Verify one segment touches the bottom edge and one touches the top edge.
        let touches_bottom = segs
            .iter()
            .any(|s| approx(s.a.y, 0.0) || approx(s.b.y, 0.0));
        let touches_top = segs
            .iter()
            .any(|s| approx(s.a.y, 1.0) || approx(s.b.y, 1.0));
        assert!(touches_bottom && touches_top);
    }

    #[test]
    fn saddle_case5_center_high_connects_across_center() {
        // v0,v2 very high so center avg exceeds iso -> alternate connection.
        let field = cell2(10.0, 0.0, 10.0, 0.0);
        let segs = extract_contours(&field, 2, 2, 0.5);
        assert_eq!(segs.len(), 2);
        // center high -> segments bottom(0)->right(1) and top(2)->left(3).
        // One segment should join bottom & right edges; another top & left.
        let has_bottom_right = segs.iter().any(|s| {
            (approx(s.a.y, 0.0) && approx(s.b.x, 1.0)) || (approx(s.b.y, 0.0) && approx(s.a.x, 1.0))
        });
        assert!(has_bottom_right);
    }

    #[test]
    fn saddle_disambiguation_changes_topology() {
        // Same corner pattern (case 5) but different center sign -> different pairing.
        let low_center = cell2(1.0, 0.0, 1.0, 0.0); // center 0.5
        let high_center = cell2(10.0, 0.0, 10.0, 0.0); // center 5.0
        let a = extract_contours(&low_center, 2, 2, 0.6);
        let b = extract_contours(&high_center, 2, 2, 0.5);
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        // The two disambiguations must not produce the identical segment set.
        let same = a
            .iter()
            .all(|sa| b.iter().any(|sb| segment_connects(*sb, sa.a, sa.b)));
        assert!(!same);
    }

    #[test]
    fn saddle_case10_produces_two_segments() {
        // v1,v3 high; v0,v2 low -> case 10 -> two segments regardless of center.
        let field = cell2(0.0, 1.0, 0.0, 1.0);
        let segs = extract_contours(&field, 2, 2, 0.5);
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn circular_field_produces_closed_contour() {
        // value = R - distance(center) -> inside disk is value>=0.
        let w = 11usize;
        let h = 11usize;
        let cx = 5.0_f32;
        let cy = 5.0_f32;
        let r = 3.0_f32;
        let mut field = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                field[y * w + x] = r - dist;
            }
        }
        let segs = extract_contours(&field, w, h, 0.0);
        assert!(
            segs.len() >= 8,
            "expected a ring of segments, got {}",
            segs.len()
        );
        // A simple closed loop: every endpoint is shared by exactly two segments,
        // so each distinct endpoint occurs an even number of times.
        let mut endpoints: Vec<Point2> = Vec::new();
        for s in &segs {
            endpoints.push(s.a);
            endpoints.push(s.b);
        }
        for p in &endpoints {
            let count = endpoints.iter().filter(|q| points_equal(**q, *p)).count();
            assert!(
                count % 2 == 0,
                "endpoint ({}, {}) shared {} times (open contour)",
                p.x,
                p.y,
                count
            );
        }
    }

    #[test]
    fn circular_contour_points_lie_near_radius() {
        // Every contour vertex should be ~R from the center for a distance field.
        let w = 13usize;
        let h = 13usize;
        let cx = 6.0_f32;
        let cy = 6.0_f32;
        let r = 4.0_f32;
        let mut field = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                field[y * w + x] = r - (dx * dx + dy * dy).sqrt();
            }
        }
        let segs = extract_contours(&field, w, h, 0.0);
        assert!(!segs.is_empty());
        let center = Point2::new(cx, cy);
        for s in &segs {
            // Linear interpolation of a nonlinear distance field is not exact,
            // but the vertices must stay within a small band around R.
            let da = center.distance(s.a);
            let db = center.distance(s.b);
            assert!((da - r).abs() < 0.5, "vertex a radius {da}");
            assert!((db - r).abs() < 0.5, "vertex b radius {db}");
        }
    }

    #[test]
    fn iso_equal_to_corner_value_boundary_behaviour() {
        // iso exactly equals the high corner value: that corner is still inside.
        let field = vec![2.0, 0.0, 0.0, 0.0];
        let segs = extract_contours(&field, 2, 2, 2.0);
        // v0 == iso -> inside (case 1), so one segment is produced.
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn non_square_grid_wide() {
        // 4 wide, 2 tall. Make only the middle-left region cross iso.
        // Row 0: 0 1 0 0 ; Row 1: 0 0 0 0
        let field = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let segs = extract_contours(&field, 4, 2, 0.5);
        // The high sample at (1,0) belongs to cells cx=0 and cx=1 (both row 0).
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn non_square_grid_tall_matches_transposed_intuition() {
        // 2 wide, 4 tall with a single high sample at (0,1).
        // Sample index: (x,y) -> y*2 + x. high at (0,1) -> index 2.
        let mut field = vec![0.0_f32; 8];
        field[2] = 1.0;
        let segs = extract_contours(&field, 2, 4, 0.5);
        // High sample (0,1) is a corner of cells (cy=0) and (cy=1). Two segments.
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn degenerate_one_by_one_is_empty() {
        let field = vec![1.0_f32];
        let segs = extract_contours(&field, 1, 1, 0.5);
        assert!(segs.is_empty());
    }

    #[test]
    fn degenerate_single_column_is_empty() {
        let field = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let segs = extract_contours(&field, 1, 5, 0.5);
        assert!(segs.is_empty());
    }

    #[test]
    fn degenerate_single_row_is_empty() {
        let field = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let segs = extract_contours(&field, 5, 1, 0.5);
        assert!(segs.is_empty());
    }

    #[test]
    fn zero_dimensions_are_empty() {
        let field: Vec<f32> = Vec::new();
        assert!(extract_contours(&field, 0, 0, 0.5).is_empty());
    }

    #[test]
    fn short_field_slice_is_empty() {
        // Claim a 3x3 grid but provide too few samples.
        let field = vec![1.0, 0.0, 0.0];
        let segs = extract_contours(&field, 3, 3, 0.5);
        assert!(segs.is_empty());
    }

    #[test]
    fn linear_field_gives_straight_vertical_contour() {
        // field value = x (column). iso 2.5 crosses only the column-2 cells.
        let w = 5usize;
        let h = 4usize;
        let mut field = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                field[y * w + x] = x as f32;
            }
        }
        let segs = extract_contours(&field, w, h, 2.5);
        assert!(!segs.is_empty());
        // Every endpoint must lie on the vertical line x = 2.5.
        for s in &segs {
            assert!(approx(s.a.x, 2.5), "a.x = {}", s.a.x);
            assert!(approx(s.b.x, 2.5), "b.x = {}", s.b.x);
        }
        // One crossing cell per row-gap: h-1 = 3 segments.
        assert_eq!(segs.len(), h - 1);
    }

    #[test]
    fn linear_field_gives_straight_horizontal_contour() {
        // field value = y (row). iso 1.5 crosses only the row-1 cells.
        let w = 4usize;
        let h = 5usize;
        let mut field = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                field[y * w + x] = y as f32;
            }
        }
        let segs = extract_contours(&field, w, h, 1.5);
        assert!(!segs.is_empty());
        for s in &segs {
            assert!(approx(s.a.y, 1.5), "a.y = {}", s.a.y);
            assert!(approx(s.b.y, 1.5), "b.y = {}", s.b.y);
        }
        assert_eq!(segs.len(), w - 1);
    }

    #[test]
    fn translation_invariance_single_peak_shifts_segments() {
        // A single high sample on a low background: its contour is a diamond
        // around the peak. Shifting the peak by (1,0) shifts every segment by
        // (1,0), because the local sample neighbourhood is identical.
        let w = 6usize;
        let h = 6usize;
        let mut field_a = vec![0.0_f32; w * h];
        let mut field_b = vec![0.0_f32; w * h];
        // Peak A at (2,2); peak B at (3,2).
        field_a[2 * w + 2] = 1.0;
        field_b[2 * w + 3] = 1.0;
        let segs_a = extract_contours(&field_a, w, h, 0.5);
        let segs_b = extract_contours(&field_b, w, h, 0.5);
        assert!(!segs_a.is_empty());
        assert_eq!(segs_a.len(), segs_b.len());
        // Each shifted-A segment must appear in B.
        for sa in &segs_a {
            let shifted_a = Point2::new(sa.a.x + 1.0, sa.a.y);
            let shifted_b = Point2::new(sa.b.x + 1.0, sa.b.y);
            let found = segs_b
                .iter()
                .any(|sb| segment_connects(*sb, shifted_a, shifted_b));
            assert!(found, "no translated match for segment");
        }
    }

    #[test]
    fn value_offset_invariance() {
        // Adding a constant k to every sample and to iso leaves geometry fixed.
        let base = vec![4.0, 0.0, 0.0, 0.0];
        let shifted: Vec<f32> = base.iter().map(|v| v + 7.0).collect();
        let a = extract_contours(&base, 2, 2, 1.0);
        let b = extract_contours(&shifted, 2, 2, 8.0);
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), 1);
        assert!(segment_connects(a[0], b[0].a, b[0].b));
    }

    #[test]
    fn segment_count_matches_number_of_crossed_cells() {
        // Two separated single-high samples far apart -> two independent diamonds.
        let w = 9usize;
        let h = 5usize;
        let mut field = vec![0.0_f32; w * h];
        field[2 * w + 2] = 1.0; // peak 1
        field[2 * w + 6] = 1.0; // peak 2, non-adjacent
        let segs = extract_contours(&field, w, h, 0.5);
        // Each isolated peak contributes 4 segments (one per surrounding cell).
        assert_eq!(segs.len(), 8);
    }

    #[test]
    fn point_lerp_interpolates_linearly() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(4.0, 8.0);
        let m = a.lerp(b, 0.25);
        assert!(point_approx(m, 1.0, 2.0));
    }

    #[test]
    fn point_add_sub_scale_roundtrip() {
        let a = Point2::new(3.0, -2.0);
        let b = Point2::new(1.5, 4.0);
        let back = a.add(b).sub(b);
        assert!(point_approx(back, 3.0, -2.0));
        let doubled = a.scale(2.0);
        assert!(point_approx(doubled, 6.0, -4.0));
    }

    #[test]
    fn segment_length_is_euclidean() {
        let s = Segment::new(Point2::new(0.0, 0.0), Point2::new(3.0, 4.0));
        assert!(approx(s.length(), 5.0));
    }

    #[test]
    fn contour_endpoints_stay_within_grid_bounds() {
        // A random-ish block; all endpoints must lie in [0,w-1] x [0,h-1].
        let w = 5usize;
        let h = 5usize;
        let mut field = vec![0.0_f32; w * h];
        // Fill a 3x3 interior block with high values.
        for y in 1..4 {
            for x in 1..4 {
                field[y * w + x] = 1.0;
            }
        }
        let segs = extract_contours(&field, w, h, 0.5);
        assert!(!segs.is_empty());
        let max_x = (w - 1) as f32;
        let max_y = (h - 1) as f32;
        for s in &segs {
            for p in [s.a, s.b] {
                assert!(p.x >= 0.0 && p.x <= max_x, "x out of bounds: {}", p.x);
                assert!(p.y >= 0.0 && p.y <= max_y, "y out of bounds: {}", p.y);
            }
        }
    }

    #[test]
    fn solid_block_forms_closed_boundary() {
        // A filled interior block yields a closed rectangular-ish contour:
        // every endpoint shared an even number of times.
        let w = 6usize;
        let h = 6usize;
        let mut field = vec![0.0_f32; w * h];
        for y in 1..5 {
            for x in 1..5 {
                field[y * w + x] = 1.0;
            }
        }
        let segs = extract_contours(&field, w, h, 0.5);
        assert!(segs.len() >= 8);
        let mut endpoints: Vec<Point2> = Vec::new();
        for s in &segs {
            endpoints.push(s.a);
            endpoints.push(s.b);
        }
        for p in &endpoints {
            let count = endpoints.iter().filter(|q| points_equal(**q, *p)).count();
            assert!(count % 2 == 0, "open boundary at ({}, {})", p.x, p.y);
        }
    }

    #[test]
    fn negative_iso_and_negative_field_values() {
        // Ensure sign handling is correct with negative thresholds.
        let field = vec![-1.0, -3.0, -3.0, -3.0];
        let segs = extract_contours(&field, 2, 2, -2.0);
        // v0=-1 >= -2 inside; others -3 < -2 outside -> case 1 -> one segment.
        assert_eq!(segs.len(), 1);
    }
}
