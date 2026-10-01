//! Analytic sub-pixel line coverage for hair anti-aliasing (design doc §8.6
//! item20).
//!
//! A single hair fibre projected to screen is almost always *thinner than one
//! pixel*: typical strand widths land well under a pixel at normal viewing
//! distance. A naive scan-line / point-sampled rasteriser either misses such a
//! fibre entirely or snaps it on and off between frames, which is the classic
//! shimmering-hair artefact. The film-grade fix (as used by `UE5` Groom's hair
//! rasteriser and the `Weta`/`Pixar` line-AA literature) is to treat each strand
//! segment not as a hard mask but as a *width-carrying line* and compute, in
//! closed form, how much of each pixel that line covers; the coverage ratio is
//! then used directly as the fibre's anti-aliasing `alpha`.
//!
//! This module is the deterministic, panic-free geometry kernel for that: given
//! one screen-space segment (a fibre segment already projected to pixel
//! coordinates) with a `width`, and a pixel centre, it returns the fraction of
//! that pixel the segment covers, in `[0, 1]`. It is a *material-independent*
//! quantity exactly like the melanin absorption map in [`crate::hair::melanin`]:
//! array in, array out, golden-comparable, no device state.
//!
//! Coverage model (simple robust variant). The pixel is approximated by a round
//! coverage kernel of radius `0.5` centred on the pixel centre (so its
//! characteristic footprint is one pixel), and the fibre segment is approximated
//! by a capsule / strip of half-width `width * 0.5`. The signed overlap of the
//! strip's half-width plus the pixel-kernel radius, minus the point-to-segment
//! distance `d`, gives a monotone trapezoidal coverage ramp:
//! `coverage = clamp(width*0.5 + 0.5 - d, 0, 1)`. This is purely closed form, has
//! no iteration, uses only `sqrt` for the distance (no transcendental math), and
//! is monotone in both `d` (closer -> more coverage) and `width` (wider -> more
//! coverage), which is exactly what a stable analytic line-AA ramp needs.
//!
//! Relation to §8.5 item12. The reactive-mask / blue-noise dither path provides
//! *stochastic* sub-pixel coverage (a dither threshold decides whether a thin
//! fibre's pixel is drawn), which is cheap but noisy. This analytic coverage is
//! the complementary *deterministic* path: it is preferred where it applies
//! (crisp, flicker-free, temporally stable thin strands) and the blue-noise
//! dither remains the fallback for cases the analytic ramp does not model (dense
//! overlapping fibres, order-independent transparency budget exhaustion). The two
//! are complementary: analytic first, dither as the safety net.

use alloc::vec::Vec;

/// A hair fibre segment already projected into screen (pixel) space, carrying a
/// stroke `width` in pixels. `width` is sanitised to a finite, non-negative value
/// by [`ScreenSegment::sanitized_width`] before use, so negative / non-finite
/// widths never panic and never produce out-of-range coverage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenSegment {
    /// Segment start in pixel coordinates.
    pub a: [f32; 2],
    /// Segment end in pixel coordinates.
    pub b: [f32; 2],
    /// Stroke width in pixels; negative / non-finite is treated as `0`.
    pub width: f32,
}

impl ScreenSegment {
    /// A screen segment from explicit endpoints and width.
    #[must_use]
    pub const fn new(a: [f32; 2], b: [f32; 2], width: f32) -> Self {
        Self { a, b, width }
    }

    /// This segment's width clamped to a finite, non-negative value (negative or
    /// non-finite -> `0`), without any floating-point equality test.
    #[must_use]
    pub fn sanitized_width(self) -> f32 {
        if self.width.is_finite() && self.width > 0.0 {
            self.width
        } else {
            0.0
        }
    }
}

/// Shortest Euclidean distance from point `p` to the segment `[a, b]` (not the
/// infinite line), with the projection parameter clamped to `[0, 1]` so both
/// endpoints are respected. A degenerate segment (`a` within [`EPS`] of `b`)
/// collapses to the point distance `|p - a|` rather than dividing by a zero
/// length, so it never panics or returns `NaN`.
#[must_use]
pub fn point_segment_distance(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let len_sq = abx * abx + aby * aby;

    let apx = p[0] - a[0];
    let apy = p[1] - a[1];

    // Degenerate segment: treat as the single point `a`.
    if len_sq.abs() < EPS {
        return (apx * apx + apy * apy).sqrt();
    }

    // Clamp the projection parameter to the segment so endpoints are respected.
    let t = ((apx * abx + apy * aby) / len_sq).clamp(0.0, 1.0);
    let cx = a[0] + t * abx;
    let cy = a[1] + t * aby;
    let dx = p[0] - cx;
    let dy = p[1] - cy;
    (dx * dx + dy * dy).sqrt()
}

/// The reference epsilon used to detect a degenerate (zero-length) segment and
/// to compare distances in tests, kept module-level so the kernel and its tests
/// agree on "close".
const EPS: f32 = 1e-6;

/// Analytic fraction of the pixel at `pixel_center` covered by `seg`, in
/// `[0, 1]`.
///
/// The pixel is modelled as a round coverage kernel of radius `0.5` and the
/// fibre as a capsule of half-width `seg.width * 0.5`; the coverage is the
/// monotone trapezoidal ramp `clamp(width*0.5 + 0.5 - d, 0, 1)`, where `d` is the
/// point-to-segment distance from the pixel centre. Closer segments and wider
/// strokes cover more; distant segments cover `0`; a segment passing through the
/// pixel centre (`d` near `0`) covers `1`. The result is always finite and in
/// range because `width` is sanitised first and the ramp is clamped.
#[must_use]
pub fn pixel_coverage(seg: ScreenSegment, pixel_center: [f32; 2]) -> f32 {
    let half_width = seg.sanitized_width() * 0.5;
    let d = point_segment_distance(pixel_center, seg.a, seg.b);
    (half_width + 0.5 - d).clamp(0.0, 1.0)
}

/// Per-pixel coverage for a whole span of pixel centres: maps each centre
/// through [`pixel_coverage`] against the same `seg`, preserving input order. An
/// empty slice returns an empty [`Vec`] (no panic); this is the array-in /
/// array-out form used to shade a tile of pixels against one fibre segment.
#[must_use]
pub fn coverage_map(seg: ScreenSegment, pixel_centers: &[[f32; 2]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pixel_centers.len());
    for &center in pixel_centers {
        out.push(pixel_coverage(seg, center));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn distance_to_point_on_segment_is_zero() {
        let d = point_segment_distance([1.0, 0.0], [0.0, 0.0], [2.0, 0.0]);
        assert!(d < EPS);
    }

    #[test]
    fn distance_clamps_to_endpoints() {
        // Point beyond `b` measures to `b`, not to the infinite line.
        let d = point_segment_distance([5.0, 0.0], [0.0, 0.0], [2.0, 0.0]);
        assert!(close(d, 3.0));
    }

    #[test]
    fn distance_perpendicular_offset() {
        let d = point_segment_distance([1.0, 2.0], [0.0, 0.0], [2.0, 0.0]);
        assert!(close(d, 2.0));
    }

    #[test]
    fn degenerate_segment_is_point_distance() {
        let d = point_segment_distance([3.0, 4.0], [0.0, 0.0], [0.0, 0.0]);
        assert!(close(d, 5.0));
    }

    #[test]
    fn center_on_segment_is_fully_covered() {
        let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 1.0);
        let coverage = pixel_coverage(seg, [5.0, 0.0]);
        assert!(close(coverage, 1.0));
    }

    #[test]
    fn far_pixel_has_zero_coverage() {
        let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 1.0);
        let coverage = pixel_coverage(seg, [5.0, 100.0]);
        assert!(close(coverage, 0.0));
    }

    #[test]
    fn coverage_is_monotone_decreasing_in_distance() {
        let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 1.0);
        let near = pixel_coverage(seg, [5.0, 0.2]);
        let mid = pixel_coverage(seg, [5.0, 0.6]);
        let far = pixel_coverage(seg, [5.0, 1.0]);
        assert!(near >= mid);
        assert!(mid >= far);
    }

    #[test]
    fn coverage_is_monotone_increasing_in_width() {
        let center = [5.0, 0.7];
        let thin = pixel_coverage(ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 0.2), center);
        let wide = pixel_coverage(ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 1.5), center);
        assert!(wide >= thin);
    }

    #[test]
    fn degenerate_segment_does_not_panic_and_is_reasonable() {
        let seg = ScreenSegment::new([2.0, 2.0], [2.0, 2.0], 1.0);
        let on_point = pixel_coverage(seg, [2.0, 2.0]);
        assert!(close(on_point, 1.0));
        let away = pixel_coverage(seg, [2.0, 100.0]);
        assert!(close(away, 0.0));
    }

    #[test]
    fn negative_and_non_finite_width_sanitize_to_zero() {
        let center = [5.0, 0.0];
        let base = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 0.0);
        let negative = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], -3.0);
        let nan = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], f32::NAN);
        let inf = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], f32::INFINITY);

        let base_cov = pixel_coverage(base, center);
        assert!(close(pixel_coverage(negative, center), base_cov));
        let nan_cov = pixel_coverage(nan, center);
        let inf_cov = pixel_coverage(inf, center);
        assert!(close(nan_cov, base_cov));
        assert!(close(inf_cov, base_cov));
    }

    #[test]
    fn coverage_stays_in_unit_range() {
        let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 3.0);
        for i in 0..200 {
            let y = (i as f32) * 0.05 - 5.0;
            let coverage = pixel_coverage(seg, [5.0, y]);
            assert!(coverage.is_finite());
            assert!(coverage >= 0.0);
            assert!(coverage <= 1.0);
        }
    }

    #[test]
    fn empty_map_is_empty_without_panic() {
        let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 1.0);
        let mapped = coverage_map(seg, &[]);
        assert!(mapped.is_empty());
    }

    #[test]
    fn map_matches_scalar_and_preserves_order() {
        let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 1.0);
        let centers = [[5.0, 0.0], [5.0, 0.5], [5.0, 2.0], [5.0, 100.0]];
        let mapped = coverage_map(seg, &centers);
        assert_eq!(mapped.len(), centers.len());
        for (center, got) in centers.iter().zip(mapped.iter()) {
            assert!(close(*got, pixel_coverage(seg, *center)));
        }
    }
}
