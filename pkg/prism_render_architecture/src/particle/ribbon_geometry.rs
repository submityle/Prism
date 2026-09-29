//! Ribbon/trail *geometry expansion*: turning an ordered centerline into the
//! left/right corner positions of a camera-facing quad strip (design §15).
//!
//! This module is the geometric complement of
//! [`super::renderers::segment_ribbons`]. That routine owns ribbon *topology*:
//! it groups [`super::renderers::RibbonSample`]s into chains, links each
//! [`super::renderers::RibbonVertex`] to its `prev`/`next` neighbour, and
//! accumulates `arc_length` and per-vertex `width` from the size-over-life
//! `LUT`. It never produces the two side corners a `GPU` build pass rasterizes.
//!
//! Here we do only that orthogonal step: given an ordered centerline, a matching
//! per-point `width`, and a camera position (or a fixed normal), emit the two
//! world-space corner points that flank each centerline point plus a
//! normalized `UV`.v. We deliberately do **not** import `renderers` (avoiding a
//! module cycle and re-doing segmentation): callers pass an already-segmented,
//! already-widthed run of points. Widths are consumed as given — the
//! size-over-life `LUT` sampling stays in `renderers`, not here.
//!
//! Only `sqrt` (through [`Vec3`]) and `floor`/`ceil`/`abs`-class arithmetic are
//! used; no transcendental functions appear, so this `CPU` reference stays
//! bit-reproducible against a future `GPU` kernel.

use alloc::vec::Vec;

use super::Vec3;

/// Absolute tolerance for the degeneracy and total-length guards in this module.
///
/// Squared comparisons use `EPS * EPS`; scalar divisions are gated on a
/// `> EPS` denominator so normalization never yields `NaN`.
pub const EPS: f32 = 1e-6;

/// The two camera-facing corner points expanded from one centerline point,
/// tagged with its normalized `UV`.v (design §15).
///
/// `left` and `right` straddle the centerline point along the strip's side
/// (binormal) axis, half a `width` out on each side. `uv_v` is the cumulative
/// arc length from the centerline head divided by the total length, in `0..=1`,
/// giving a length-parameterized `V` coordinate for texturing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RibbonStripVertex {
    /// World-space corner on the `+side` (binormal) half of the strip.
    pub left: Vec3,
    /// World-space corner on the `-side` (binormal) half of the strip.
    pub right: Vec3,
    /// Normalized arc-length coordinate along the strip, in `0..=1`.
    pub uv_v: f32,
}

/// Expands a centerline into a *camera-facing* ribbon strip (design §15).
///
/// For each centerline point the tangent is estimated by central difference
/// (`next - prev`) at interior points and by a one-sided difference at the
/// endpoints, then normalized. The view direction is `camera_position - point`;
/// the side (binormal) axis is `tangent x view`, normalized, which keeps the
/// strip's flat face turned toward the camera. When that cross product
/// degenerates (tangent nearly parallel to the view ray, or a zero tangent from
/// coincident points) a deterministic fallback axis perpendicular to the
/// tangent is used, so no `NaN` is produced. The corners are
/// `point +/- side * (0.5 * width)`.
///
/// `widths` need not match `centerline` in length: index `i` uses `widths[i]`
/// when present, otherwise the last supplied width, otherwise `1.0`; negative
/// widths are clamped to `0.0`. An empty centerline yields an empty strip; a
/// single point yields a single (safe, finite) vertex.
#[must_use]
pub fn build_strip(
    centerline: &[Vec3],
    widths: &[f32],
    camera_position: Vec3,
) -> Vec<RibbonStripVertex> {
    build_with(centerline, widths, |point, tangent| {
        let view = camera_position.sub(point);
        camera_facing_side(tangent, view)
    })
}

/// Expands a centerline into a ribbon strip about a *fixed* normal (design §15).
///
/// Identical to [`build_strip`] except the side (binormal) axis is
/// `tangent x fixed_normal` instead of being derived from the camera. This is
/// the fixed-axis ribbon variant: the strip does not turn to face the camera,
/// which is what beam-like or world-locked trails want. The same deterministic
/// fallback and width/length rules as [`build_strip`] apply.
#[must_use]
pub fn build_strip_flat(
    centerline: &[Vec3],
    widths: &[f32],
    fixed_normal: Vec3,
) -> Vec<RibbonStripVertex> {
    build_with(centerline, widths, |_point, tangent| {
        let raw = tangent.cross(fixed_normal).normalize_or_zero();
        if raw.length_squared() > EPS * EPS {
            raw
        } else {
            fallback_side(tangent)
        }
    })
}

/// Builds the triangle-list indices for a strip of `vertex_count` centerline
/// vertices (design §15).
///
/// The corner vertices produced by [`build_strip`] are assumed uploaded
/// interleaved: centerline point `i` occupies `left` at `GPU` index `2*i` and
/// `right` at `2*i + 1`. Every span between consecutive points emits two
/// triangles with alternating left/right winding, so the result has
/// `6 * (vertex_count - 1)` indices (empty when `vertex_count < 2`). Callers
/// that prefer a `GPU` `triangle-strip` topology can instead upload the corners
/// in order and skip this list entirely.
#[must_use]
pub fn strip_indices(vertex_count: u32) -> Vec<u32> {
    let mut indices = Vec::new();
    if vertex_count < 2 {
        return indices;
    }
    for i in 0..(vertex_count - 1) {
        let left0 = 2 * i;
        let right0 = 2 * i + 1;
        let left1 = 2 * (i + 1);
        let right1 = 2 * (i + 1) + 1;
        // First triangle covers the left edge of the span, second the right.
        indices.push(left0);
        indices.push(right0);
        indices.push(left1);
        indices.push(right0);
        indices.push(right1);
        indices.push(left1);
    }
    indices
}

/// Shared expansion core: resolves tangents, per-point widths, and normalized
/// arc-length `UV`.v, deferring the side (binormal) axis to `side_of`.
fn build_with(
    centerline: &[Vec3],
    widths: &[f32],
    mut side_of: impl FnMut(Vec3, Vec3) -> Vec3,
) -> Vec<RibbonStripVertex> {
    let count = centerline.len();
    let mut strip = Vec::new();
    if count == 0 {
        return strip;
    }

    // Cumulative arc length along the centerline, for the length-parameterized
    // `UV`.v. The head is always 0; the total gates the division below.
    let mut cumulative = Vec::with_capacity(count);
    cumulative.push(0.0_f32);
    for i in 1..count {
        let span = centerline[i].distance(centerline[i - 1]);
        cumulative.push(cumulative[i - 1] + span);
    }
    let total = cumulative[count - 1];
    let has_length = total > EPS;

    for i in 0..count {
        let point = centerline[i];
        let tangent = tangent_at(centerline, i);
        let side = side_of(point, tangent);
        let half = 0.5 * resolve_width(widths, i);
        let offset = side.scale(half);
        let uv_v = if has_length {
            (cumulative[i] / total).clamp(0.0, 1.0)
        } else {
            0.0
        };
        strip.push(RibbonStripVertex {
            left: point.add(offset),
            right: point.sub(offset),
            uv_v,
        });
    }

    strip
}

/// Estimates the unit tangent at centerline index `i`: central difference at
/// interior points, one-sided at the endpoints, zero for a lone point.
fn tangent_at(centerline: &[Vec3], i: usize) -> Vec3 {
    let count = centerline.len();
    if count < 2 {
        return Vec3::ZERO;
    }
    let raw = if i == 0 {
        centerline[1].sub(centerline[0])
    } else if i + 1 == count {
        centerline[count - 1].sub(centerline[count - 2])
    } else {
        centerline[i + 1].sub(centerline[i - 1])
    };
    raw.normalize_or_zero()
}

/// Camera-facing side (binormal) axis `tangent x view`, with a deterministic
/// fallback when the two are (near) parallel or the tangent is zero.
fn camera_facing_side(tangent: Vec3, view: Vec3) -> Vec3 {
    let raw = tangent.cross(view).normalize_or_zero();
    if raw.length_squared() > EPS * EPS {
        raw
    } else {
        fallback_side(tangent)
    }
}

/// A deterministic unit vector perpendicular to `tangent`, used when the
/// primary side axis degenerates. Falls back to the world `X` axis when the
/// tangent itself is (numerically) zero.
fn fallback_side(tangent: Vec3) -> Vec3 {
    // Cross with whichever cardinal axis is least parallel to the tangent.
    let reference = if tangent.x.abs() <= 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let side = tangent.cross(reference).normalize_or_zero();
    if side.length_squared() > EPS * EPS {
        side
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    }
}

/// Resolves the strip width at index `i`: `widths[i]` when present, else the
/// last supplied width, else `1.0`; the result is clamped to be non-negative.
fn resolve_width(widths: &[f32], i: usize) -> f32 {
    let raw = if i < widths.len() {
        widths[i]
    } else if let Some(&last) = widths.last() {
        last
    } else {
        1.0
    };
    raw.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Loose equality helper honouring the module tolerance.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4
    }

    fn all_finite(v: Vec3) -> bool {
        v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
    }

    #[test]
    fn straight_line_corners_symmetric_and_width_spaced() {
        let centerline = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let widths = vec![2.0, 2.0, 2.0];
        let camera = Vec3::new(0.0, 0.0, 5.0);
        let strip = build_strip(&centerline, &widths, camera);
        assert_eq!(strip.len(), 3);
        for (i, v) in strip.iter().enumerate() {
            // Midpoint of the two corners is the centerline point.
            let mid = v.left.add(v.right).scale(0.5);
            assert!(approx(mid.x, centerline[i].x));
            assert!(approx(mid.y, centerline[i].y));
            assert!(approx(mid.z, centerline[i].z));
            // Corner spacing equals the requested width.
            assert!(approx(v.left.distance(v.right), 2.0));
        }
    }

    #[test]
    fn camera_facing_side_perpendicular_to_tangent_and_view() {
        let centerline = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let widths = vec![2.0, 2.0, 2.0];
        let camera = Vec3::new(1.0, 0.0, 5.0);
        let strip = build_strip(&centerline, &widths, camera);
        // Interior point 1: tangent is +X, view is straight up in Z.
        let v = strip[1];
        let side = v.left.sub(v.right).normalize_or_zero();
        let tangent = Vec3::new(1.0, 0.0, 0.0);
        let view = camera.sub(centerline[1]);
        assert!(approx(side.dot(tangent), 0.0));
        assert!(approx(side.dot(view), 0.0));
    }

    #[test]
    fn uv_v_monotonic_zero_to_one() {
        let centerline = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let widths = vec![1.0, 1.0, 1.0];
        let strip = build_strip(&centerline, &widths, Vec3::new(0.0, 0.0, 1.0));
        assert!(approx(strip[0].uv_v, 0.0));
        assert!(approx(strip[2].uv_v, 1.0));
        assert!(strip[0].uv_v <= strip[1].uv_v);
        assert!(strip[1].uv_v <= strip[2].uv_v);
        // Cumulative arc length at the middle knot is 1 of 3 total units -> 1/3.
        assert!(approx(strip[1].uv_v, 1.0 / 3.0));
    }

    #[test]
    fn fixed_normal_variant_uses_supplied_axis() {
        let centerline = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let widths = vec![2.0, 2.0];
        let strip = build_strip_flat(&centerline, &widths, Vec3::new(0.0, 0.0, 1.0));
        // tangent +X crossed with normal +Z gives the -Y side axis.
        let side = strip[0].left.sub(strip[0].right).normalize_or_zero();
        assert!(approx(side.x, 0.0));
        assert!(approx(side.y.abs(), 1.0));
        assert!(approx(side.z, 0.0));
        assert!(approx(strip[0].left.distance(strip[0].right), 2.0));
    }

    #[test]
    fn coincident_points_produce_no_nan() {
        let centerline = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 2.0, 3.0),
        ];
        let widths = vec![1.0, 1.0, 1.0];
        let strip = build_strip(&centerline, &widths, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(strip.len(), 3);
        for v in &strip {
            assert!(all_finite(v.left));
            assert!(all_finite(v.right));
            assert!(v.uv_v.is_finite());
            // Zero total length collapses every `UV`.v to 0.
            assert!(approx(v.uv_v, 0.0));
        }
    }

    #[test]
    fn zero_width_collapses_corners() {
        let centerline = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let widths = vec![0.0, 0.0];
        let strip = build_strip(&centerline, &widths, Vec3::new(0.0, 0.0, 1.0));
        for v in &strip {
            assert!(approx(v.left.distance(v.right), 0.0));
            assert!(all_finite(v.left));
        }
    }

    #[test]
    fn single_point_is_safe() {
        let centerline = vec![Vec3::new(4.0, 5.0, 6.0)];
        let widths = vec![2.0];
        let strip = build_strip(&centerline, &widths, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(strip.len(), 1);
        assert!(all_finite(strip[0].left));
        assert!(all_finite(strip[0].right));
        assert!(approx(strip[0].uv_v, 0.0));
    }

    #[test]
    fn empty_centerline_is_empty() {
        let strip = build_strip(&[], &[], Vec3::new(0.0, 0.0, 1.0));
        assert!(strip.is_empty());
    }

    #[test]
    fn width_length_mismatch_falls_back() {
        let centerline = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        // Shorter than centerline: index 2 reuses the last width (3.0).
        let widths = vec![1.0, 3.0];
        let strip = build_strip(&centerline, &widths, Vec3::new(0.0, 0.0, 1.0));
        assert!(approx(strip[0].left.distance(strip[0].right), 1.0));
        assert!(approx(strip[2].left.distance(strip[2].right), 3.0));
        // Empty widths default to unit width.
        let unit = build_strip(&centerline, &[], Vec3::new(0.0, 0.0, 1.0));
        assert!(approx(unit[1].left.distance(unit[1].right), 1.0));
    }

    #[test]
    fn strip_indices_count_and_layout() {
        assert!(strip_indices(0).is_empty());
        assert!(strip_indices(1).is_empty());
        let four = strip_indices(4);
        // 3 spans * 6 indices per span.
        assert_eq!(four.len(), 18);
        // First span references the first two corner pairs.
        assert_eq!(&four[0..6], &[0, 1, 2, 1, 3, 2]);
    }
}
