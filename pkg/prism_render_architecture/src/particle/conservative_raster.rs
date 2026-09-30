//! Conservative triangle rasterization primitives for the particle
//! decal-projection and mesh-emission reference paths (design §8.2, §16).
//!
//! A *standard* rasterizer covers a pixel only when the pixel *center* falls
//! inside the triangle. A *conservative* rasterizer instead covers every pixel
//! the triangle *touches at all*, which the particle subsystem needs whenever a
//! sparse footprint must not drop thin slivers: decal splatting onto a surface,
//! trail-ribbon coverage, and any `GPU` bin/tile assignment that must not miss a
//! partially-covered tile. This module owns the small, `CPU`-verifiable contract
//! those paths share: the screen-space edge equations of a triangle, the
//! half-pixel outward *dilation* that turns a center test into a
//! touch-anything test, the conservative integer bounding box (an `AABB` with a
//! one-pixel safety ring), and the per-pixel coverage predicate.
//!
//! # Winding
//! Edges are always emitted with a counter-clockwise (`CCW`) interior: a
//! clockwise (`CW`) input triangle is normalized by swapping two corners so the
//! coverage predicate is a single "all edge values non-negative" test
//! regardless of the caller's winding.
//!
//! # No transcendental math
//! Every routine is pure `+`, `-`, `*`, comparison, `f32::floor`, `f32::abs`,
//! `f32::min`, `f32::max` and `f32::clamp`. There is no `sin`, `sqrt`, `atan`,
//! or any transcendental call, and no `f32::ceil`: rounding up is done as
//! `-((-x).floor())` so the two rounding directions share one primitive. A
//! degenerate (zero-area) triangle can never produce coverage: its area is
//! checked against [`CMP_EPS`] and reported as empty.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};
use alloc::vec::Vec;

/// Magnitude below which a signed area or an edge value is treated as zero.
///
/// Floats are never compared with `==`/`!=`; a value is "zero" when its
/// absolute value is below this threshold, and a pixel is "covered" by an edge
/// when its edge value is at least `-CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

/// Byte size of one `std430`-packed [`Edge`] record, reusing the shared
/// `vec4<f32>` stride: three coefficients plus one word of tail padding so the
/// record aligns like a `GPU` `vec4`.
pub const CONSERVATIVE_RASTER_STD430_SIZE: usize = VEC4_STRIDE;

/// A screen-space edge line `a*x + b*y + c = 0`.
///
/// For a normalized counter-clockwise (`CCW`) triangle the gradient `(a, b)`
/// points *into* the triangle, so [`Edge::eval`] is non-negative exactly on the
/// interior side of the edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    /// `x` coefficient of the line equation.
    pub a: f32,
    /// `y` coefficient of the line equation.
    pub b: f32,
    /// Constant term of the line equation.
    pub c: f32,
}

impl Edge {
    /// Evaluates the signed edge value `a*x + b*y + c` at `(x, y)`.
    ///
    /// The result is proportional to the signed distance from the line (scaled
    /// by the un-normalized normal length `hypot(a, b)`), and is non-negative on
    /// the interior side of a normalized `CCW` edge.
    #[must_use]
    pub fn eval(&self, x: f32, y: f32) -> f32 {
        self.a * x + self.b * y + self.c
    }

    /// Packs the edge into a `std430`-compatible little-endian byte record.
    ///
    /// Layout: `a` at bytes `0..4`, `b` at `4..8`, `c` at `8..12`, and a final
    /// zero word at `12..16` so the record occupies a full `vec4<f32>` slot.
    #[must_use]
    pub fn to_std430(&self) -> [u8; CONSERVATIVE_RASTER_STD430_SIZE] {
        let mut bytes = [0u8; CONSERVATIVE_RASTER_STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.a.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.b.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.c.to_le_bytes());
        bytes
    }
}

/// An inclusive integer bounding box in pixel coordinates.
///
/// The box is a conservative `AABB`: it always contains a one-pixel safety ring
/// around the triangle and is clamped to the screen, so iterating
/// `min_x..=max_x` by `min_y..=max_y` visits every pixel the triangle can
/// possibly touch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AabbI {
    /// Smallest pixel column in the box (inclusive).
    pub min_x: i32,
    /// Smallest pixel row in the box (inclusive).
    pub min_y: i32,
    /// Largest pixel column in the box (inclusive).
    pub max_x: i32,
    /// Largest pixel row in the box (inclusive).
    pub max_y: i32,
}

/// Twice the signed area of the triangle `v`.
///
/// The sign encodes the winding: positive for counter-clockwise (`CCW`),
/// negative for clockwise (`CW`), and near zero for a degenerate triangle.
#[must_use]
pub fn triangle_area2(v: [[f32; 2]; 3]) -> f32 {
    (v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0])
}

/// Returns `true` when the triangle has effectively zero area (its three
/// corners are collinear or coincident) and therefore cannot cover any pixel.
#[must_use]
pub fn is_degenerate(v: [[f32; 2]; 3]) -> bool {
    triangle_area2(v).abs() < CMP_EPS
}

/// Builds the directed edge line for the segment from `p0` to `p1`.
///
/// The gradient `(a, b)` is the inward normal for a `CCW` triangle, so the edge
/// value is non-negative to the left of `p0 -> p1`.
fn edge_between(p0: [f32; 2], p1: [f32; 2]) -> Edge {
    let a = p0[1] - p1[1];
    let b = p1[0] - p0[0];
    let c = (p1[1] - p0[1]) * p0[0] - (p1[0] - p0[0]) * p0[1];
    Edge { a, b, c }
}

/// Returns the three directed edges of the triangle `v`, normalized to a
/// counter-clockwise (`CCW`) interior.
///
/// A clockwise (`CW`) input is normalized by swapping its last two corners, so
/// the interior is always the side where all three edge values are
/// non-negative regardless of the caller's winding.
#[must_use]
pub fn edges_from_triangle(v: [[f32; 2]; 3]) -> [Edge; 3] {
    let ccw = if triangle_area2(v) < 0.0 {
        [v[0], v[2], v[1]]
    } else {
        v
    };
    [
        edge_between(ccw[0], ccw[1]),
        edge_between(ccw[1], ccw[2]),
        edge_between(ccw[2], ccw[0]),
    ]
}

/// Pushes a single edge outward along its normal by `half_pixel * (|a| + |b|)`.
fn dilate_one(e: Edge, half_pixel: f32) -> Edge {
    Edge {
        a: e.a,
        b: e.b,
        c: e.c + half_pixel * (e.a.abs() + e.b.abs()),
    }
}

/// Dilates all three edges outward by `half_pixel * (|a| + |b|)` each.
///
/// Because the edge value equals `hypot(a, b)` times the signed distance from
/// the line, adding `half_pixel * (|a| + |b|)` shifts the line out by the
/// maximum projection of a `half_pixel`-radius pixel box onto the edge normal.
/// A `half_pixel` of `0.5` therefore guarantees that any pixel the triangle
/// touches has a non-negative dilated edge value at its center: the
/// center-inside test on the dilated edges is a touch-anything test on the
/// original triangle.
#[must_use]
pub fn dilate_edges(edges: &[Edge; 3], half_pixel: f32) -> [Edge; 3] {
    [
        dilate_one(edges[0], half_pixel),
        dilate_one(edges[1], half_pixel),
        dilate_one(edges[2], half_pixel),
    ]
}

/// Floors `x` to the nearest integer at or below it, as an `i32`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "x is floored to an integral f32 and float-to-int casts saturate, so no wrapping truncation occurs"
)]
fn floor_to_i32(x: f32) -> i32 {
    x.floor() as i32
}

/// Ceils `x` to the nearest integer at or above it, as an `i32`.
///
/// Rounding up is expressed as `-((-x).floor())` to avoid `f32::ceil`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the ceil value is an integral f32 and float-to-int casts saturate, so no wrapping truncation occurs"
)]
fn ceil_to_i32(x: f32) -> i32 {
    let ceil = -((-x).floor());
    ceil as i32
}

/// Computes the conservative integer bounding box of the triangle `v` on a
/// `pixels_w` by `pixels_h` screen.
///
/// The raw box is `floor(min) - 1 ..= ceil(max) + 1` on each axis (a one-pixel
/// safety ring so a triangle grazing a pixel boundary is never missed), then
/// clamped into `0 ..= pixels_w - 1` and `0 ..= pixels_h - 1`. An off-screen
/// triangle clamps to the nearest screen edge.
#[must_use]
pub fn conservative_bounds(v: [[f32; 2]; 3], pixels_w: i32, pixels_h: i32) -> AabbI {
    let min_x = v[0][0].min(v[1][0]).min(v[2][0]);
    let max_x = v[0][0].max(v[1][0]).max(v[2][0]);
    let min_y = v[0][1].min(v[1][1]).min(v[2][1]);
    let max_y = v[0][1].max(v[1][1]).max(v[2][1]);

    let hi_x = (pixels_w - 1).max(0);
    let hi_y = (pixels_h - 1).max(0);

    AabbI {
        min_x: (floor_to_i32(min_x) - 1).clamp(0, hi_x),
        min_y: (floor_to_i32(min_y) - 1).clamp(0, hi_y),
        max_x: (ceil_to_i32(max_x) + 1).clamp(0, hi_x),
        max_y: (ceil_to_i32(max_y) + 1).clamp(0, hi_y),
    }
}

/// Returns the center `(px + 0.5, py + 0.5)` of pixel `(px, py)`.
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel indices are small screen coordinates, exactly representable in f32"
)]
fn pixel_center(px: i32, py: i32) -> [f32; 2] {
    [px as f32 + 0.5, py as f32 + 0.5]
}

/// Returns `true` when the center of pixel `(px, py)` lies on the interior side
/// of all three (typically dilated) edges.
///
/// A pixel is covered when every edge value at its center is at least
/// `-CMP_EPS`; the epsilon slack keeps pixels whose centers land exactly on an
/// edge from being dropped to floating-point noise.
#[must_use]
pub fn pixel_covered(edges: &[Edge; 3], px: i32, py: i32) -> bool {
    let [cx, cy] = pixel_center(px, py);
    edges.iter().all(|e| e.eval(cx, cy) >= -CMP_EPS)
}

/// Rasterizes the conservative coverage of triangle `v` on a `w` by `h` screen,
/// returning every pixel coordinate the triangle touches.
///
/// The edges are normalized to `CCW`, dilated by `half_pixel`, and tested at
/// each pixel center inside the conservative bounding box. A degenerate
/// triangle yields no coverage. Use `half_pixel = 0.5` for true conservative
/// coverage and `half_pixel = 0.0` for a standard center-inside rasterization.
#[must_use]
pub fn rasterize_coverage(v: [[f32; 2]; 3], w: i32, h: i32, half_pixel: f32) -> Vec<(i32, i32)> {
    if is_degenerate(v) {
        return Vec::new();
    }
    let bounds = conservative_bounds(v, w, h);
    let edges = dilate_edges(&edges_from_triangle(v), half_pixel);
    let (min_x, max_x) = (bounds.min_x, bounds.max_x);
    let (min_y, max_y) = (bounds.min_y, bounds.max_y);
    (min_y..=max_y)
        .flat_map(move |py| (min_x..=max_x).map(move |px| (px, py)))
        .filter(|&(px, py)| pixel_covered(&edges, px, py))
        .collect()
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`Edge`] records.
///
/// Reuses [`CONSERVATIVE_RASTER_STD430_SIZE`] and the shared clamp-to-one
/// element rule from [`storage_bytes`], so an empty set still yields a valid
/// non-empty `GPU` binding.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(CONSERVATIVE_RASTER_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CCW_UNIT: [[f32; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    const CW_UNIT: [[f32; 2]; 3] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]];

    #[test]
    fn edge_eval_matches_line_equation() {
        let e = Edge {
            a: 2.0,
            b: -3.0,
            c: 1.0,
        };
        assert!((e.eval(1.0, 1.0) - 0.0).abs() < CMP_EPS);
        assert!((e.eval(2.0, 0.0) - 5.0).abs() < CMP_EPS);
    }

    #[test]
    fn triangle_area2_sign_encodes_winding() {
        assert!(triangle_area2(CCW_UNIT) > 0.0);
        assert!(triangle_area2(CW_UNIT) < 0.0);
        assert!((triangle_area2(CCW_UNIT).abs() - 1.0).abs() < CMP_EPS);
    }

    #[test]
    fn degenerate_collinear_is_detected() {
        let line = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        assert!(is_degenerate(line));
        let coincident = [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]];
        assert!(is_degenerate(coincident));
    }

    #[test]
    fn non_degenerate_triangle_is_not_flagged() {
        assert!(!is_degenerate(CCW_UNIT));
        assert!(!is_degenerate(CW_UNIT));
    }

    #[test]
    fn ccw_interior_point_is_non_negative_on_all_edges() {
        let edges = edges_from_triangle(CCW_UNIT);
        let inside = [0.25_f32, 0.25_f32];
        for e in &edges {
            assert!(e.eval(inside[0], inside[1]) >= -CMP_EPS);
        }
    }

    #[test]
    fn cw_input_is_normalized_to_ccw() {
        let edges = edges_from_triangle(CW_UNIT);
        let inside = [0.25_f32, 0.25_f32];
        // Despite CW input the interior test is still "all non-negative".
        for e in &edges {
            assert!(e.eval(inside[0], inside[1]) >= -CMP_EPS);
        }
    }

    #[test]
    fn exterior_point_is_negative_on_some_edge() {
        let edges = edges_from_triangle(CCW_UNIT);
        let outside = [2.0_f32, 2.0_f32];
        let any_negative = edges.iter().any(|e| e.eval(outside[0], outside[1]) < 0.0);
        assert!(any_negative);
    }

    #[test]
    fn edge_gradient_points_inward() {
        // For a CCW triangle the gradient (a, b) of each edge should point
        // toward the centroid: moving from a boundary point toward the
        // centroid must not decrease the edge value.
        let edges = edges_from_triangle(CCW_UNIT);
        let centroid = [1.0_f32 / 3.0, 1.0_f32 / 3.0];
        for e in &edges {
            assert!(e.eval(centroid[0], centroid[1]) >= -CMP_EPS);
        }
    }

    #[test]
    fn dilate_shifts_constant_by_offset() {
        let edges = edges_from_triangle(CCW_UNIT);
        let half = 0.5_f32;
        let dilated = dilate_edges(&edges, half);
        for (raw, dil) in edges.iter().zip(dilated.iter()) {
            let offset = half * (raw.a.abs() + raw.b.abs());
            assert!((dil.c - raw.c - offset).abs() < CMP_EPS);
            assert!((dil.a - raw.a).abs() < CMP_EPS);
            assert!((dil.b - raw.b).abs() < CMP_EPS);
        }
    }

    #[test]
    fn dilate_with_zero_half_pixel_is_identity() {
        let edges = edges_from_triangle(CCW_UNIT);
        let dilated = dilate_edges(&edges, 0.0);
        for (raw, dil) in edges.iter().zip(dilated.iter()) {
            assert!((dil.c - raw.c).abs() < CMP_EPS);
        }
    }

    #[test]
    fn dilated_edges_cover_more_pixels_than_standard() {
        let v = [[2.0, 2.0], [6.0, 2.0], [2.0, 6.0]];
        let strict = rasterize_coverage(v, 12, 12, 0.0);
        let conservative = rasterize_coverage(v, 12, 12, 0.5);
        assert!(conservative.len() > strict.len());
    }

    #[test]
    fn conservative_coverage_is_superset_of_standard() {
        let v = [[2.0, 2.0], [6.0, 2.0], [2.0, 6.0]];
        let strict = rasterize_coverage(v, 12, 12, 0.0);
        let conservative = rasterize_coverage(v, 12, 12, 0.5);
        for pixel in &strict {
            assert!(conservative.contains(pixel));
        }
    }

    #[test]
    fn conservative_bounds_includes_one_pixel_ring() {
        let v = [[2.0, 3.0], [7.0, 3.0], [2.0, 8.0]];
        let b = conservative_bounds(v, 100, 100);
        assert_eq!(b.min_x, 1);
        assert_eq!(b.min_y, 2);
        assert_eq!(b.max_x, 8);
        assert_eq!(b.max_y, 9);
    }

    #[test]
    fn conservative_bounds_clamped_to_screen_high() {
        let v = [[2.0, 3.0], [7.0, 3.0], [2.0, 8.0]];
        let b = conservative_bounds(v, 5, 5);
        assert_eq!(b.max_x, 4);
        assert_eq!(b.max_y, 4);
        assert_eq!(b.min_x, 1);
        assert_eq!(b.min_y, 2);
    }

    #[test]
    fn conservative_bounds_clamped_to_screen_low() {
        let v = [[-5.0, -5.0], [-1.0, -5.0], [-5.0, -1.0]];
        let b = conservative_bounds(v, 16, 16);
        assert_eq!(b.min_x, 0);
        assert_eq!(b.min_y, 0);
        assert_eq!(b.max_x, 0);
        assert_eq!(b.max_y, 0);
    }

    #[test]
    fn conservative_bounds_handles_fractional_corners() {
        let v = [[2.4, 3.7], [7.1, 3.2], [2.9, 8.6]];
        let b = conservative_bounds(v, 100, 100);
        // floor(2.4) - 1 = 1, floor(3.2) - 1 = 2.
        assert_eq!(b.min_x, 1);
        assert_eq!(b.min_y, 2);
        // ceil(7.1) + 1 = 9, ceil(8.6) + 1 = 10.
        assert_eq!(b.max_x, 9);
        assert_eq!(b.max_y, 10);
    }

    #[test]
    fn pixel_center_covered_when_inside() {
        let edges = edges_from_triangle([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]);
        let dilated = dilate_edges(&edges, 0.5);
        assert!(pixel_covered(&dilated, 2, 2));
    }

    #[test]
    fn pixel_center_not_covered_when_far_outside() {
        let edges = edges_from_triangle([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]);
        let dilated = dilate_edges(&edges, 0.5);
        assert!(!pixel_covered(&dilated, 20, 20));
    }

    #[test]
    fn fully_interior_pixels_are_covered() {
        let v = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let coverage = rasterize_coverage(v, 16, 16, 0.5);
        // These centers are strictly inside the triangle (x + y < 10).
        assert!(coverage.contains(&(1, 1)));
        assert!(coverage.contains(&(2, 2)));
        assert!(coverage.contains(&(3, 1)));
    }

    #[test]
    fn rasterize_degenerate_triangle_is_empty() {
        let line = [[0.0, 0.0], [4.0, 4.0], [8.0, 8.0]];
        assert!(rasterize_coverage(line, 16, 16, 0.5).is_empty());
    }

    #[test]
    fn rasterize_nonempty_for_real_triangle() {
        let v = [[1.0, 1.0], [5.0, 1.0], [1.0, 5.0]];
        assert!(!rasterize_coverage(v, 16, 16, 0.5).is_empty());
    }

    #[test]
    fn rasterize_ccw_and_cw_agree() {
        let ccw = [[1.0, 1.0], [5.0, 1.0], [1.0, 5.0]];
        let cw = [[1.0, 1.0], [1.0, 5.0], [5.0, 1.0]];
        let mut a = rasterize_coverage(ccw, 16, 16, 0.5);
        let mut b = rasterize_coverage(cw, 16, 16, 0.5);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn coverage_pixels_lie_within_bounds() {
        let v = [[2.0, 2.0], [9.0, 3.0], [3.0, 9.0]];
        let b = conservative_bounds(v, 32, 32);
        for &(px, py) in &rasterize_coverage(v, 32, 32, 0.5) {
            assert!((b.min_x..=b.max_x).contains(&px));
            assert!((b.min_y..=b.max_y).contains(&py));
        }
    }

    #[test]
    fn std430_size_matches_vec4_stride() {
        assert_eq!(CONSERVATIVE_RASTER_STD430_SIZE, VEC4_STRIDE);
        assert_eq!(CONSERVATIVE_RASTER_STD430_SIZE, 16);
    }

    #[test]
    fn std430_roundtrip_decodes_coefficients() {
        let e = Edge {
            a: 1.5,
            b: -2.25,
            c: 3.75,
        };
        let bytes = e.to_std430();
        assert_eq!(bytes.len(), CONSERVATIVE_RASTER_STD430_SIZE);
        let da = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let db = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let dc = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert!((da - e.a).abs() < CMP_EPS);
        assert!((db - e.b).abs() < CMP_EPS);
        assert!((dc - e.c).abs() < CMP_EPS);
    }

    #[test]
    fn std430_tail_word_is_zero_padding() {
        let e = Edge {
            a: 9.0,
            b: 8.0,
            c: 7.0,
        };
        let bytes = e.to_std430();
        assert_eq!(&bytes[12..16], &[0u8; 4]);
    }

    #[test]
    fn gpu_storage_bytes_reuses_shared_rule() {
        assert_eq!(gpu_storage_bytes(0), CONSERVATIVE_RASTER_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(4), 4 * CONSERVATIVE_RASTER_STD430_SIZE);
        assert_eq!(
            gpu_storage_bytes(10),
            storage_bytes(CONSERVATIVE_RASTER_STD430_SIZE, 10)
        );
    }
}
