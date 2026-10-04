//! Arrow (caret) positioning for anchored overlays.
//!
//! A popover or tooltip usually paints a small triangular *arrow* on the edge
//! facing its anchor, so the surface visually "points at" the trigger. After
//! [`position`](crate::position) has placed (and possibly flipped and shifted)
//! the surface, the arrow must slide along the surface's cross axis to stay
//! aligned with the anchor's centre — yet never slide past the surface's
//! rounded corners. This is the `arrow` middleware from the Popper / Floating UI
//! model, expressed as pure geometry over [`prism_ui_layout`] rectangles.
//!
//! The arrow lives on the cross axis of the chosen [`Side`](crate::Side):
//! horizontal for [`Top`](crate::Side::Top) / [`Bottom`](crate::Side::Bottom)
//! placements, vertical for [`Left`](crate::Side::Left) /
//! [`Right`](crate::Side::Right). Its ideal centre is the anchor's cross-axis
//! centre, clamped into the surface inset by `padding` (typically the surface's
//! corner radius) on each end so the caret never detaches from a flat edge.
//!
//! # Example
//!
//! ```
//! use prism_ui_layout::{Point, Rect, Size};
//! use prism_ui_overlay::{arrow, position, ArrowConfig, Placement, PositionConfig, Side};
//!
//! let anchor = Rect::new(Point::new(100.0, 100.0), Size::new(60.0, 20.0));
//! let floating = Size::new(140.0, 48.0);
//! let viewport = Rect::new(Point::new(0.0, 0.0), Size::new(2000.0, 2000.0));
//!
//! let config = PositionConfig {
//!     placement: Placement::new(Side::Bottom, prism_ui_overlay::Align::Center),
//!     offset: 8.0,
//!     ..PositionConfig::default()
//! };
//! let placed = position(anchor, floating, viewport, &config);
//!
//! // A 12px arrow with 6px of corner inset.
//! let caret = arrow(anchor, &placed, &ArrowConfig::new(12.0).with_padding(6.0));
//!
//! // Bottom placement => the caret slides along x and sits under the anchor's
//! // horizontal centre (130), well away from the surface corners here.
//! assert!(caret.cross_horizontal);
//! assert!(!caret.clamped);
//! assert!((caret.center - 130.0).abs() < 1e-3);
//! ```

use crate::position::Positioned;
use prism_ui_layout::Rect;

/// Inputs to [`arrow`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ArrowConfig {
    /// Extent of the arrow along the surface's cross axis.
    pub size: f32,
    /// Minimum gap kept between the arrow and each cross-axis corner of the
    /// surface, usually the surface's border radius.
    pub padding: f32,
}

impl ArrowConfig {
    /// A zero-padding arrow of the given cross-axis extent.
    #[must_use]
    pub const fn new(size: f32) -> Self {
        Self { size, padding: 0.0 }
    }

    /// Returns a copy of this config with the given corner `padding`.
    #[must_use]
    pub const fn with_padding(self, padding: f32) -> Self {
        Self {
            size: self.size,
            padding,
        }
    }
}

/// The resolved position of an overlay's arrow along the surface cross axis.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ArrowPosition {
    /// Distance from the surface's leading cross-axis edge (its left for a
    /// horizontal cross axis, its top for a vertical one) to the arrow's
    /// leading edge.
    pub offset: f32,
    /// Absolute cross-axis coordinate of the arrow's centre.
    pub center: f32,
    /// Whether the cross axis is horizontal (`true` for
    /// [`Top`](crate::Side::Top) / [`Bottom`](crate::Side::Bottom) placements).
    pub cross_horizontal: bool,
    /// Whether the ideal centre had to be clamped to keep the arrow inside the
    /// padded surface (the caret could not reach the anchor's centre).
    pub clamped: bool,
}

/// Positions an overlay's arrow so it points at `anchor` while staying inside
/// the already-placed surface described by `placed`.
///
/// The arrow slides only along the surface's cross axis. Its ideal centre is
/// the anchor's cross-axis centre; that centre is clamped so the arrow keeps at
/// least `config.padding` from each cross-axis corner. When the surface is too
/// small to host the arrow plus padding the arrow is centred and
/// [`clamped`](ArrowPosition::clamped) is set.
#[must_use]
pub fn arrow(anchor: Rect, placed: &Positioned, config: &ArrowConfig) -> ArrowPosition {
    let surface = placed.rect;
    // Side's main axis is horizontal for Left/Right, so the *cross* axis is
    // horizontal exactly when the placement side is vertical (Top/Bottom).
    let cross_horizontal = !placed.placement.side.is_horizontal();

    let (cross_start, cross_len, anchor_center) = if cross_horizontal {
        (
            surface.left(),
            surface.size.width,
            anchor.left() + anchor.size.width / 2.0,
        )
    } else {
        (
            surface.top(),
            surface.size.height,
            anchor.top() + anchor.size.height / 2.0,
        )
    };

    let ideal_offset = anchor_center - cross_start - config.size / 2.0;
    let lo = config.padding;
    let hi = cross_len - config.size - config.padding;

    let (offset, clamped) = if hi < lo {
        // The surface cannot host the arrow plus its padding: centre the arrow
        // and report that we could not honour the ideal position.
        ((cross_len - config.size) / 2.0, true)
    } else {
        let clamped_offset = ideal_offset.clamp(lo, hi);
        (
            clamped_offset,
            (clamped_offset - ideal_offset).abs() > f32::EPSILON,
        )
    };

    ArrowPosition {
        offset,
        center: cross_start + offset + config.size / 2.0,
        cross_horizontal,
        clamped,
    }
}

#[cfg(test)]
mod tests {
    use super::{arrow, ArrowConfig};
    use crate::position::{position, Align, Placement, PositionConfig, Side};
    use prism_ui_layout::{Point, Rect, Size};

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            let t = (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32);
            lo + t * (hi - lo)
        }
    }

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::new(Point::new(x, y), Size::new(w, h))
    }

    fn pick_side(n: u64) -> Side {
        match n % 4 {
            0 => Side::Top,
            1 => Side::Right,
            2 => Side::Bottom,
            _ => Side::Left,
        }
    }

    fn pick_align(n: u64) -> Align {
        match n % 3 {
            0 => Align::Start,
            1 => Align::Center,
            _ => Align::End,
        }
    }

    #[test]
    fn matches_clamped_center_oracle() {
        let mut rng = SplitMix64(0x5EED_1234);
        for _ in 0..5000 {
            let viewport = rect(0.0, 0.0, 2000.0, 2000.0);
            let anchor = rect(
                rng.range(50.0, 1500.0),
                rng.range(50.0, 1500.0),
                rng.range(4.0, 120.0),
                rng.range(4.0, 120.0),
            );
            let floating = Size::new(rng.range(20.0, 400.0), rng.range(20.0, 400.0));
            let side = pick_side(rng.next_u64());
            let config = PositionConfig {
                placement: Placement::new(side, pick_align(rng.next_u64())),
                offset: rng.range(0.0, 20.0),
                align_offset: rng.range(-40.0, 40.0),
                flip: true,
                shift: true,
                padding: 0.0,
            };
            let placed = position(anchor, floating, viewport, &config);

            let size = rng.range(4.0, 40.0);
            let padding = rng.range(0.0, 20.0);
            let acfg = ArrowConfig::new(size).with_padding(padding);
            let got = arrow(anchor, &placed, &acfg);

            // Independent oracle derived from the *final* placed surface.
            let final_side = placed.placement.side;
            let cross_horizontal = !final_side.is_horizontal();
            assert_eq!(got.cross_horizontal, cross_horizontal);
            let (cross_start, cross_len, anchor_center) = if cross_horizontal {
                (
                    placed.rect.left(),
                    placed.rect.size.width,
                    anchor.left() + anchor.size.width / 2.0,
                )
            } else {
                (
                    placed.rect.top(),
                    placed.rect.size.height,
                    anchor.top() + anchor.size.height / 2.0,
                )
            };
            let ideal = anchor_center - cross_start - size / 2.0;
            let lo = padding;
            let hi = cross_len - size - padding;
            let (exp_offset, exp_clamped) = if hi < lo {
                ((cross_len - size) / 2.0, true)
            } else {
                let c = ideal.clamp(lo, hi);
                (c, (c - ideal).abs() > f32::EPSILON)
            };
            assert!((got.offset - exp_offset).abs() < 1e-3, "offset");
            assert_eq!(got.clamped, exp_clamped, "clamped");
            assert!(
                (got.center - (cross_start + exp_offset + size / 2.0)).abs() < 1e-3,
                "center",
            );

            // When the surface can host the arrow, it stays within the padded
            // span and its centre is consistent with its offset.
            if hi >= lo {
                assert!(got.offset >= lo - 1e-3);
                assert!(got.offset + size <= cross_len - padding + 1e-3);
            }
            assert!((got.center - (cross_start + got.offset + size / 2.0)).abs() < 1e-3);

            // Determinism.
            let again = arrow(anchor, &placed, &acfg);
            assert_eq!(got, again);
        }
    }

    #[test]
    fn points_at_anchor_center_when_unclamped() {
        // Large surface centred over the anchor: the caret should reach the
        // anchor's centre exactly and not be clamped.
        let viewport = rect(0.0, 0.0, 2000.0, 2000.0);
        let anchor = rect(500.0, 500.0, 40.0, 20.0);
        let floating = Size::new(300.0, 120.0);
        let config = PositionConfig {
            placement: Placement::new(Side::Bottom, Align::Center),
            offset: 10.0,
            flip: false,
            shift: false,
            ..PositionConfig::default()
        };
        let placed = position(anchor, floating, viewport, &config);
        let got = arrow(anchor, &placed, &ArrowConfig::new(12.0).with_padding(8.0));
        assert!(got.cross_horizontal);
        assert!(!got.clamped);
        assert!((got.center - 520.0).abs() < 1e-3); // anchor centre x = 500 + 40/2
    }

    #[test]
    fn clamps_toward_corner_for_edge_anchor() {
        use crate::position::Positioned;
        // Surface at x=0..200; anchor centre far to the right (410) cannot be
        // reached, so the arrow pins to the trailing padded corner.
        let placed = Positioned {
            rect: rect(0.0, 540.0, 200.0, 80.0),
            placement: Placement::new(Side::Bottom, Align::Center),
            shifted: false,
            fits: true,
        };
        let anchor = rect(400.0, 500.0, 20.0, 20.0); // centre x = 410
        let size = 10.0;
        let padding = 5.0;
        let got = arrow(anchor, &placed, &ArrowConfig::new(size).with_padding(padding));
        assert!(got.clamped);
        // Pinned to the trailing bound (right corner).
        let hi = 200.0 - size - padding;
        assert!((got.offset - hi).abs() < 1e-3);

        // Mirror case: anchor centre far to the left pins to the leading corner.
        let anchor_left = rect(-60.0, 500.0, 20.0, 20.0); // centre x = -50
        let got_left = arrow(anchor_left, &placed, &ArrowConfig::new(size).with_padding(padding));
        assert!(got_left.clamped);
        assert!((got_left.offset - padding).abs() < 1e-3);
    }

    #[test]
    fn degenerate_surface_centers_arrow() {
        let viewport = rect(0.0, 0.0, 2000.0, 2000.0);
        let anchor = rect(500.0, 500.0, 20.0, 20.0);
        // Surface narrower than arrow + 2*padding on the cross axis.
        let floating = Size::new(14.0, 60.0);
        let config = PositionConfig {
            placement: Placement::new(Side::Bottom, Align::Center),
            offset: 4.0,
            flip: false,
            shift: false,
            ..PositionConfig::default()
        };
        let placed = position(anchor, floating, viewport, &config);
        let size = 12.0;
        let padding = 6.0; // 12 + 12 = 24 > 14 => degenerate
        let got = arrow(anchor, &placed, &ArrowConfig::new(size).with_padding(padding));
        assert!(got.clamped);
        assert!((got.offset - (placed.rect.size.width - size) / 2.0).abs() < 1e-3);
    }

    #[test]
    fn vertical_cross_axis_for_side_placement() {
        let viewport = rect(0.0, 0.0, 2000.0, 2000.0);
        let anchor = rect(500.0, 500.0, 20.0, 40.0);
        let floating = Size::new(120.0, 300.0);
        let config = PositionConfig {
            placement: Placement::new(Side::Right, Align::Center),
            offset: 8.0,
            flip: false,
            shift: false,
            ..PositionConfig::default()
        };
        let placed = position(anchor, floating, viewport, &config);
        let got = arrow(anchor, &placed, &ArrowConfig::new(10.0).with_padding(4.0));
        assert!(!got.cross_horizontal);
        assert!(!got.clamped);
        // Anchor centre y = 500 + 40/2 = 520.
        assert!((got.center - 520.0).abs() < 1e-3);
    }
}
