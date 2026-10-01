//! Screen-space probe placement.
//!
//! Lumen-style diffuse GI seeds one radiance probe per screen tile.  This
//! module maps between the framebuffer pixel grid and the coarse probe grid:
//! how many probes a viewport needs, which pixel rectangle each probe owns, and
//! where each probe's centre lands.
//!
//! # Conventions
//! * Pixel coordinates use the graphics convention: `x` grows right, `y` grows
//!   down, with pixel `(0, 0)` at the top-left corner.  A pixel *centre* is at
//!   integer coordinate `+ 0.5`.
//! * Probes tile the viewport left-to-right then top-to-bottom.  Probe
//!   `(px, py)` owns the pixel rectangle starting at `(px * tile, py * tile)`.
//! * Edge tiles are clamped to the viewport, so the right/bottom probes may own
//!   a narrower rectangle when the viewport is not an exact multiple of the
//!   tile size.  A probe's centre is the centre of its (clamped) rectangle, so
//!   edge probes bias toward the covered pixels rather than off-screen area.

use bevy_math::{UVec2, Vec2};

/// Integer pixel rectangle owned by a single probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelRect {
    /// Inclusive top-left pixel of the rectangle.
    pub min: UVec2,
    /// Rectangle extent in pixels (`width`, `height`).  A zero component means
    /// the probe covers no pixels (degenerate viewport or out-of-range probe).
    pub size: UVec2,
}

impl PixelRect {
    /// The empty rectangle.
    pub const EMPTY: Self = Self {
        min: UVec2::ZERO,
        size: UVec2::ZERO,
    };

    /// One-past-the-last pixel (exclusive) on each axis.
    #[inline]
    pub fn max_exclusive(&self) -> UVec2 {
        self.min + self.size
    }

    /// Number of pixels the rectangle covers.
    #[inline]
    pub fn area(&self) -> u32 {
        self.size.x * self.size.y
    }

    /// Sub-pixel centre of the rectangle (pixel-centre convention).  Returns
    /// `None` for an empty rectangle, which has no meaningful centre.
    #[inline]
    pub fn center(&self) -> Option<Vec2> {
        if self.size.x == 0 || self.size.y == 0 {
            return None;
        }
        Some(Vec2::new(
            self.min.x as f32 + self.size.x as f32 * 0.5,
            self.min.y as f32 + self.size.y as f32 * 0.5,
        ))
    }
}

/// Clamps the tile size to at least one pixel so callers can never divide by
/// zero or request an infinite probe count.
#[inline]
fn sanitized_tile(tile: u32) -> u32 {
    tile.max(1)
}

/// Ceil-division helper for non-negative integers.
#[inline]
fn div_ceil(value: u32, divisor: u32) -> u32 {
    debug_assert!(divisor > 0);
    value.div_ceil(divisor)
}

/// Number of probes along each axis for `viewport` at `tile` pixels per probe.
///
/// A zero-sized viewport yields zero probes on that axis; a `tile` of zero is
/// treated as one pixel.
#[inline]
pub fn probe_grid_dims(viewport: UVec2, tile: u32) -> UVec2 {
    let tile = sanitized_tile(tile);
    UVec2::new(div_ceil(viewport.x, tile), div_ceil(viewport.y, tile))
}

/// Total probe count for `viewport` at `tile` pixels per probe.
#[inline]
pub fn probe_count(viewport: UVec2, tile: u32) -> u32 {
    let dims = probe_grid_dims(viewport, tile);
    dims.x * dims.y
}

/// Row-major flattened index of probe `(px, py)` given the grid `dims`.
///
/// Returns `None` when the probe is outside the grid.
#[inline]
pub fn probe_index(probe: UVec2, dims: UVec2) -> Option<u32> {
    if probe.x >= dims.x || probe.y >= dims.y {
        return None;
    }
    Some(probe.y * dims.x + probe.x)
}

/// Inverse of [`probe_index`]: recovers `(px, py)` from a flattened index.
///
/// Returns `None` when `index` is out of range for `dims`.
#[inline]
pub fn probe_coord(index: u32, dims: UVec2) -> Option<UVec2> {
    if dims.x == 0 || index >= dims.x * dims.y {
        return None;
    }
    Some(UVec2::new(index % dims.x, index / dims.x))
}

/// Pixel rectangle owned by probe `(px, py)`, clamped to the viewport.
///
/// Out-of-range probes and degenerate viewports return [`PixelRect::EMPTY`].
#[inline]
pub fn probe_pixel_rect(probe: UVec2, viewport: UVec2, tile: u32) -> PixelRect {
    let tile = sanitized_tile(tile);
    let dims = probe_grid_dims(viewport, tile);
    if probe.x >= dims.x || probe.y >= dims.y {
        return PixelRect::EMPTY;
    }
    let min = UVec2::new(probe.x * tile, probe.y * tile);
    // The tile may run past the viewport on the last row/column; clamp it.
    let max_x = ((probe.x + 1) * tile).min(viewport.x);
    let max_y = ((probe.y + 1) * tile).min(viewport.y);
    let size = UVec2::new(max_x.saturating_sub(min.x), max_y.saturating_sub(min.y));
    PixelRect { min, size }
}

/// Sub-pixel centre of probe `(px, py)`.
///
/// Returns `None` when the probe owns no pixels.
#[inline]
pub fn probe_center_pixel(probe: UVec2, viewport: UVec2, tile: u32) -> Option<Vec2> {
    probe_pixel_rect(probe, viewport, tile).center()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_multiple_grid() {
        let vp = UVec2::new(64, 32);
        assert_eq!(probe_grid_dims(vp, 8), UVec2::new(8, 4));
        assert_eq!(probe_count(vp, 8), 32);
    }

    #[test]
    fn non_divisible_edges_round_up() {
        let vp = UVec2::new(70, 34);
        // ceil(70/8)=9, ceil(34/8)=5
        assert_eq!(probe_grid_dims(vp, 8), UVec2::new(9, 5));
        assert_eq!(probe_count(vp, 8), 45);
    }

    #[test]
    fn edge_tile_is_clamped() {
        let vp = UVec2::new(70, 34);
        // Last column probe (index 8) starts at x=64, viewport width 70 -> 6px.
        let rect = probe_pixel_rect(UVec2::new(8, 0), vp, 8);
        assert_eq!(rect.min, UVec2::new(64, 0));
        assert_eq!(rect.size, UVec2::new(6, 8));
        assert_eq!(rect.area(), 48);
        // Last row probe (index 4) starts at y=32 -> 2px tall.
        let rect = probe_pixel_rect(UVec2::new(0, 4), vp, 8);
        assert_eq!(rect.size, UVec2::new(8, 2));
    }

    #[test]
    fn interior_tile_full_size_and_centre() {
        let vp = UVec2::new(64, 64);
        let rect = probe_pixel_rect(UVec2::new(2, 3), vp, 8);
        assert_eq!(rect.min, UVec2::new(16, 24));
        assert_eq!(rect.size, UVec2::new(8, 8));
        assert_eq!(rect.center(), Some(Vec2::new(20.0, 28.0)));
    }

    #[test]
    fn corner_probe_centre_biases_into_view() {
        let vp = UVec2::new(70, 34);
        // 6x2 rectangle at (64,32) -> centre (67, 33).
        let c = probe_center_pixel(UVec2::new(8, 4), vp, 8).unwrap();
        assert_eq!(c, Vec2::new(64.0 + 3.0, 32.0 + 1.0));
    }

    #[test]
    fn zero_viewport_has_no_probes() {
        assert_eq!(probe_grid_dims(UVec2::ZERO, 8), UVec2::ZERO);
        assert_eq!(probe_count(UVec2::ZERO, 8), 0);
        assert_eq!(
            probe_pixel_rect(UVec2::ZERO, UVec2::ZERO, 8),
            PixelRect::EMPTY
        );
        assert_eq!(probe_center_pixel(UVec2::ZERO, UVec2::ZERO, 8), None);
    }

    #[test]
    fn zero_tile_treated_as_one() {
        let vp = UVec2::new(4, 3);
        assert_eq!(probe_grid_dims(vp, 0), UVec2::new(4, 3));
        let rect = probe_pixel_rect(UVec2::new(1, 1), vp, 0);
        assert_eq!(rect.size, UVec2::new(1, 1));
    }

    #[test]
    fn out_of_range_probe_is_empty() {
        let vp = UVec2::new(64, 64);
        let rect = probe_pixel_rect(UVec2::new(99, 0), vp, 8);
        assert_eq!(rect, PixelRect::EMPTY);
        assert_eq!(rect.center(), None);
    }

    #[test]
    fn index_round_trip() {
        let dims = UVec2::new(9, 5);
        for i in 0..dims.x * dims.y {
            let coord = probe_coord(i, dims).unwrap();
            assert_eq!(probe_index(coord, dims), Some(i));
        }
        assert_eq!(probe_index(UVec2::new(9, 0), dims), None);
        assert_eq!(probe_coord(45, dims), None);
    }

    #[test]
    fn rectangles_tile_the_viewport_without_gaps() {
        let vp = UVec2::new(70, 34);
        let tile = 8;
        let dims = probe_grid_dims(vp, tile);
        let mut covered = 0u32;
        for py in 0..dims.y {
            for px in 0..dims.x {
                covered += probe_pixel_rect(UVec2::new(px, py), vp, tile).area();
            }
        }
        assert_eq!(covered, vp.x * vp.y);
    }
}
