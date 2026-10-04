//! Anchor positioning for popovers, tooltips and context menus.
//!
//! An [`OverlayKind::Popover`](crate::OverlayKind::Popover) or
//! [`OverlayKind::Tooltip`](crate::OverlayKind::Tooltip) is *anchored* to a
//! trigger widget: it must appear next to that widget, yet stay fully visible
//! inside the viewport. This module turns an anchor rectangle, a floating
//! surface size and a boundary rectangle into a concrete placed rectangle,
//! using the same middleware model popularised by Popper / Floating UI:
//!
//! 1. **place** the surface on the requested [`Side`] of the anchor, aligned by
//!    the requested [`Align`], separated by a main-axis `offset`;
//! 2. **flip** to the opposite side when the requested side overflows the
//!    boundary and the opposite side has more room;
//! 3. **shift** the surface along the cross axis so it stays within the
//!    boundary (clamped, never detaching from the anchor's axis).
//!
//! The engine is pure geometry over [`prism_ui_layout`] primitives, so it is
//! `no_std` friendly, deterministic and independent of any renderer.
//!
//! # Example
//!
//! ```
//! use prism_ui_layout::{Point, Rect, Size};
//! use prism_ui_overlay::{position, Align, PositionConfig, Placement, Side};
//!
//! let anchor = Rect::new(Point::new(100.0, 100.0), Size::new(60.0, 20.0));
//! let floating = Size::new(140.0, 48.0);
//! let viewport = Rect::new(Point::new(0.0, 0.0), Size::new(800.0, 600.0));
//!
//! let config = PositionConfig {
//!     placement: Placement::new(Side::Bottom, Align::Start),
//!     offset: 8.0,
//!     ..PositionConfig::default()
//! };
//! let placed = position(anchor, floating, viewport, &config);
//!
//! // Bottom/Start places the surface flush under the anchor's left edge.
//! assert_eq!(placed.placement.side, Side::Bottom);
//! assert!(placed.fits);
//! assert!((placed.rect.top() - 128.0).abs() < 1e-3); // 100 + 20 + 8
//! assert!((placed.rect.left() - 100.0).abs() < 1e-3);
//! ```

use prism_ui_layout::{Point, Rect, Size};

/// Tolerance used when deciding whether a placed surface fully fits.
///
/// Coordinates run to a few thousand logical pixels, where the `f32` rounding
/// error already exceeds [`f32::EPSILON`]; `1e-3` of a pixel is visually
/// irrelevant yet comfortably above that accumulated error.
const FIT_EPSILON: f32 = 1e-3;

/// The side of the anchor a floating surface is placed on.
///
/// The *main axis* is the axis the surface is pushed along: vertical for
/// [`Top`](Side::Top) / [`Bottom`](Side::Bottom), horizontal for
/// [`Left`](Side::Left) / [`Right`](Side::Right). The remaining axis is the
/// *cross axis* that [`Align`] and the shift step operate on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Side {
    /// Above the anchor.
    Top,
    /// To the right of the anchor.
    Right,
    /// Below the anchor.
    Bottom,
    /// To the left of the anchor.
    Left,
}

impl Side {
    /// The side directly opposite this one, used by the flip step.
    #[must_use]
    pub fn opposite(self) -> Self {
        match self {
            Side::Top => Side::Bottom,
            Side::Bottom => Side::Top,
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        }
    }

    /// Whether this side's main axis is horizontal.
    ///
    /// `true` for [`Left`](Side::Left) and [`Right`](Side::Right); the cross
    /// axis is then vertical.
    #[must_use]
    pub fn is_horizontal(self) -> bool {
        matches!(self, Side::Left | Side::Right)
    }
}

/// Alignment of the surface along the anchor's cross axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Align {
    /// Flush with the anchor's leading cross edge (left or top).
    Start,
    /// Centred on the anchor's cross extent.
    Center,
    /// Flush with the anchor's trailing cross edge (right or bottom).
    End,
}

/// A placement request: a [`Side`] plus a cross-axis [`Align`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Placement {
    /// The anchor side the surface is placed on.
    pub side: Side,
    /// The cross-axis alignment.
    pub align: Align,
}

impl Placement {
    /// Builds a placement from a side and alignment.
    #[must_use]
    pub const fn new(side: Side, align: Align) -> Self {
        Self { side, align }
    }

    /// Returns a copy of this placement with a different [`Align`].
    #[must_use]
    pub const fn with_align(self, align: Align) -> Self {
        Self {
            side: self.side,
            align,
        }
    }

    /// Returns a copy of this placement on the opposite [`Side`].
    #[must_use]
    pub fn flipped(self) -> Self {
        Self {
            side: self.side.opposite(),
            align: self.align,
        }
    }

    /// Centre-aligned placement above the anchor.
    pub const TOP: Self = Self::new(Side::Top, Align::Center);
    /// Centre-aligned placement below the anchor.
    pub const BOTTOM: Self = Self::new(Side::Bottom, Align::Center);
    /// Centre-aligned placement left of the anchor.
    pub const LEFT: Self = Self::new(Side::Left, Align::Center);
    /// Centre-aligned placement right of the anchor.
    pub const RIGHT: Self = Self::new(Side::Right, Align::Center);
}

impl Default for Placement {
    fn default() -> Self {
        Self::BOTTOM
    }
}

/// Inputs to [`position`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PositionConfig {
    /// The preferred placement before any flipping.
    pub placement: Placement,
    /// Gap along the main axis between the anchor edge and the surface.
    pub offset: f32,
    /// Extra skidding applied along the cross axis before shifting.
    pub align_offset: f32,
    /// Enables the flip step (switch to the opposite side on overflow).
    pub flip: bool,
    /// Enables the shift step (clamp along the cross axis into the boundary).
    pub shift: bool,
    /// Minimum distance kept between the surface and the boundary edges while
    /// shifting.
    pub padding: f32,
}

impl Default for PositionConfig {
    fn default() -> Self {
        Self {
            placement: Placement::default(),
            offset: 0.0,
            align_offset: 0.0,
            flip: true,
            shift: true,
            padding: 0.0,
        }
    }
}

/// The result of [`position`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Positioned {
    /// The placed rectangle of the floating surface.
    pub rect: Rect,
    /// The placement actually used (after any flip).
    pub placement: Placement,
    /// Whether the shift step moved the surface along its cross axis.
    pub shifted: bool,
    /// Whether the surface ends up fully inside the boundary.
    pub fits: bool,
}

/// Places a floating surface of size `floating` next to `anchor`, kept inside
/// `boundary`, according to `config`.
///
/// The returned [`Positioned`] carries the final rectangle, the placement
/// chosen after an optional flip, whether a shift occurred and whether the
/// surface fits entirely within the boundary.
#[must_use]
pub fn position(
    anchor: Rect,
    floating: Size<f32>,
    boundary: Rect,
    config: &PositionConfig,
) -> Positioned {
    let preferred = config.placement.side;

    // Flip: choose whichever of the preferred / opposite side overflows the
    // main axis least. Ties keep the preferred side (deterministic).
    let side = if config.flip {
        let preferred_overflow = main_overflow(preferred, anchor, floating, boundary, config.offset);
        let opposite = preferred.opposite();
        let opposite_overflow = main_overflow(opposite, anchor, floating, boundary, config.offset);
        if opposite_overflow < preferred_overflow {
            opposite
        } else {
            preferred
        }
    } else {
        preferred
    };

    let placement = Placement::new(side, config.placement.align);
    let origin = anchored_origin(
        placement,
        anchor,
        floating,
        config.offset,
        config.align_offset,
    );
    let mut x = origin.x;
    let mut y = origin.y;

    // Shift: clamp along the cross axis so the surface stays inside the
    // boundary without leaving the anchor's main axis.
    let mut shifted = false;
    if config.shift {
        if side.is_horizontal() {
            let lo = boundary.top() + config.padding;
            let hi = boundary.bottom() - config.padding - floating.height;
            let clamped = clamp_range(y, lo, hi);
            shifted = (clamped - y).abs() > f32::EPSILON;
            y = clamped;
        } else {
            let lo = boundary.left() + config.padding;
            let hi = boundary.right() - config.padding - floating.width;
            let clamped = clamp_range(x, lo, hi);
            shifted = (clamped - x).abs() > f32::EPSILON;
            x = clamped;
        }
    }

    let rect = Rect::new(Point::new(x, y), floating);
    let fits = is_zero(axis_overflow(
        x,
        floating.width,
        boundary.left(),
        boundary.right(),
    )) && is_zero(axis_overflow(
        y,
        floating.height,
        boundary.top(),
        boundary.bottom(),
    ));

    Positioned {
        rect,
        placement,
        shifted,
        fits,
    }
}

/// Computes the surface origin for a placement, before any shifting.
fn anchored_origin(
    placement: Placement,
    anchor: Rect,
    floating: Size<f32>,
    offset: f32,
    align_offset: f32,
) -> Point<f32> {
    match placement.side {
        Side::Top => Point::new(
            cross_horizontal(placement.align, anchor, floating.width) + align_offset,
            anchor.top() - floating.height - offset,
        ),
        Side::Bottom => Point::new(
            cross_horizontal(placement.align, anchor, floating.width) + align_offset,
            anchor.bottom() + offset,
        ),
        Side::Left => Point::new(
            anchor.left() - floating.width - offset,
            cross_vertical(placement.align, anchor, floating.height) + align_offset,
        ),
        Side::Right => Point::new(
            anchor.right() + offset,
            cross_vertical(placement.align, anchor, floating.height) + align_offset,
        ),
    }
}

/// Cross-axis coordinate for a horizontal cross axis (sides Top / Bottom).
fn cross_horizontal(align: Align, anchor: Rect, width: f32) -> f32 {
    match align {
        Align::Start => anchor.left(),
        Align::Center => anchor.left() + (anchor.size.width - width) / 2.0,
        Align::End => anchor.right() - width,
    }
}

/// Cross-axis coordinate for a vertical cross axis (sides Left / Right).
fn cross_vertical(align: Align, anchor: Rect, height: f32) -> f32 {
    match align {
        Align::Start => anchor.top(),
        Align::Center => anchor.top() + (anchor.size.height - height) / 2.0,
        Align::End => anchor.bottom() - height,
    }
}

/// Main-axis overflow of a candidate side (how far the surface would spill out
/// of the boundary along the side's main axis).
fn main_overflow(side: Side, anchor: Rect, floating: Size<f32>, boundary: Rect, offset: f32) -> f32 {
    let origin = anchored_origin(Placement::new(side, Align::Start), anchor, floating, offset, 0.0);
    if side.is_horizontal() {
        axis_overflow(origin.x, floating.width, boundary.left(), boundary.right())
    } else {
        axis_overflow(origin.y, floating.height, boundary.top(), boundary.bottom())
    }
}

/// How far the span `[start, start + extent]` lies outside `[min, max]`.
fn axis_overflow(start: f32, extent: f32, min: f32, max: f32) -> f32 {
    let end = start + extent;
    let before = (min - start).max(0.0);
    let after = (end - max).max(0.0);
    before + after
}

/// Clamps `v` into `[lo, hi]`, degrading to `lo` when the surface is larger
/// than the available range (`hi < lo`).
fn clamp_range(v: f32, lo: f32, hi: f32) -> f32 {
    if hi < lo {
        lo
    } else {
        v.clamp(lo, hi)
    }
}

/// Whether `x` is within [`FIT_EPSILON`] of zero.
fn is_zero(x: f32) -> bool {
    x.abs() < FIT_EPSILON
}

#[cfg(test)]
mod tests {
    use super::{
        position, Align, Placement, PositionConfig, Side,
    };
    use prism_ui_layout::{Point, Rect, Size};

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::new(Point::new(x, y), Size::new(w, h))
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn offset_and_start_alignment() {
        let anchor = rect(100.0, 100.0, 60.0, 20.0);
        let floating = Size::new(140.0, 48.0);
        let viewport = rect(0.0, 0.0, 2000.0, 2000.0);
        let config = PositionConfig {
            placement: Placement::new(Side::Bottom, Align::Start),
            offset: 8.0,
            ..PositionConfig::default()
        };
        let placed = position(anchor, floating, viewport, &config);
        assert_eq!(placed.placement.side, Side::Bottom);
        assert!(placed.fits);
        assert!(!placed.shifted);
        assert!(approx(placed.rect.top(), 128.0));
        assert!(approx(placed.rect.left(), 100.0));
    }

    #[test]
    fn center_alignment_centers_on_anchor() {
        let anchor = rect(100.0, 100.0, 60.0, 20.0);
        let floating = Size::new(140.0, 48.0);
        let viewport = rect(0.0, 0.0, 2000.0, 2000.0);
        let config = PositionConfig {
            placement: Placement::new(Side::Bottom, Align::Center),
            offset: 0.0,
            ..PositionConfig::default()
        };
        let placed = position(anchor, floating, viewport, &config);
        // Anchor centre x = 130; floating width 140 => left = 130 - 70 = 60.
        assert!(approx(placed.rect.left(), 60.0));
    }

    #[test]
    fn flips_to_opposite_side_when_no_room() {
        // Anchor pinned to the bottom edge: no room below, plenty above.
        let viewport = rect(0.0, 0.0, 400.0, 300.0);
        let anchor = rect(100.0, 280.0, 60.0, 20.0);
        let floating = Size::new(80.0, 100.0);
        let config = PositionConfig {
            placement: Placement::new(Side::Bottom, Align::Start),
            offset: 4.0,
            ..PositionConfig::default()
        };
        let placed = position(anchor, floating, viewport, &config);
        assert_eq!(placed.placement.side, Side::Top);
        assert!(placed.fits);
        // Placed above the anchor: bottom edge sits at anchor.top - offset.
        assert!(approx(placed.rect.bottom(), anchor.top() - 4.0));
    }

    #[test]
    fn no_flip_when_disabled_even_if_overflowing() {
        let viewport = rect(0.0, 0.0, 400.0, 300.0);
        let anchor = rect(100.0, 280.0, 60.0, 20.0);
        let floating = Size::new(80.0, 100.0);
        let config = PositionConfig {
            placement: Placement::new(Side::Bottom, Align::Start),
            offset: 4.0,
            flip: false,
            shift: false,
            ..PositionConfig::default()
        };
        let placed = position(anchor, floating, viewport, &config);
        assert_eq!(placed.placement.side, Side::Bottom);
        assert!(!placed.fits);
    }

    #[test]
    fn shift_clamps_into_boundary() {
        // Anchor near the right edge; surface would overflow right without shift.
        let viewport = rect(0.0, 0.0, 400.0, 400.0);
        let anchor = rect(360.0, 100.0, 30.0, 20.0);
        let floating = Size::new(160.0, 40.0);
        let config = PositionConfig {
            placement: Placement::new(Side::Bottom, Align::Start),
            offset: 6.0,
            padding: 8.0,
            ..PositionConfig::default()
        };
        let placed = position(anchor, floating, viewport, &config);
        assert!(placed.shifted);
        // Clamped so the right edge sits at boundary.right - padding.
        assert!(approx(placed.rect.right(), 400.0 - 8.0));
        assert!(placed.rect.left() >= 8.0 - 1e-3);
    }

    #[test]
    fn shift_degrades_gracefully_when_surface_larger_than_boundary() {
        let viewport = rect(0.0, 0.0, 100.0, 400.0);
        let anchor = rect(40.0, 100.0, 20.0, 20.0);
        let floating = Size::new(200.0, 40.0); // wider than the 100px viewport
        let config = PositionConfig {
            placement: Placement::new(Side::Bottom, Align::Center),
            padding: 4.0,
            ..PositionConfig::default()
        };
        let placed = position(anchor, floating, viewport, &config);
        // Cannot fit horizontally, but clamps to the padded left edge.
        assert!(approx(placed.rect.left(), 4.0));
        assert!(!placed.fits);
    }

    // -- randomized oracle ---------------------------------------------------

    struct Rng(u32);

    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }

        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            let t = (self.next_u32() as f32) / (u32::MAX as f32);
            lo + t * (hi - lo)
        }
    }

    fn pick_side(n: u32) -> Side {
        match n % 4 {
            0 => Side::Top,
            1 => Side::Right,
            2 => Side::Bottom,
            _ => Side::Left,
        }
    }

    fn pick_align(n: u32) -> Align {
        match n % 3 {
            0 => Align::Start,
            1 => Align::Center,
            _ => Align::End,
        }
    }

    #[test]
    fn property_fits_when_a_side_and_cross_axis_have_room() {
        let mut rng = Rng(0x1234_5678);
        for _ in 0..4000 {
            let bw = rng.range(100.0, 1200.0);
            let bh = rng.range(100.0, 1200.0);
            let boundary = rect(rng.range(-50.0, 50.0), rng.range(-50.0, 50.0), bw, bh);

            // Anchor fully inside the boundary.
            let aw = rng.range(4.0, bw * 0.4);
            let ah = rng.range(4.0, bh * 0.4);
            let ax = rng.range(boundary.left(), boundary.right() - aw);
            let ay = rng.range(boundary.top(), boundary.bottom() - ah);
            let anchor = rect(ax, ay, aw, ah);

            let floating = Size::new(rng.range(4.0, bw), rng.range(4.0, bh));
            let offset = rng.range(0.0, 20.0);
            let side = pick_side(rng.next_u32());
            let align = pick_align(rng.next_u32());

            let config = PositionConfig {
                placement: Placement::new(side, align),
                offset,
                align_offset: rng.range(-30.0, 30.0),
                flip: true,
                shift: true,
                padding: 0.0,
            };
            let placed = position(anchor, floating, boundary, &config);

            // Independent oracle for whether a fit is achievable.
            let (space_before, space_after, floating_main, floating_cross, boundary_cross) =
                if side.is_horizontal() {
                    (
                        anchor.left() - boundary.left(),
                        boundary.right() - anchor.right(),
                        floating.width,
                        floating.height,
                        boundary.size.height,
                    )
                } else {
                    (
                        anchor.top() - boundary.top(),
                        boundary.bottom() - anchor.bottom(),
                        floating.height,
                        floating.width,
                        boundary.size.width,
                    )
                };
            let main_fits = floating_main + offset <= space_before.max(space_after) + 1e-3;
            let cross_fits = floating_cross <= boundary_cross + 1e-3;

            if main_fits && cross_fits {
                assert!(
                    placed.fits,
                    "expected fit: side={side:?} align={align:?} \
                     before={space_before} after={space_after} \
                     fmain={floating_main} off={offset}",
                );
            }

            // Determinism: identical inputs give identical outputs.
            let again = position(anchor, floating, boundary, &config);
            assert_eq!(placed, again);
        }
    }

    #[test]
    fn property_shift_keeps_cross_axis_within_padded_boundary() {
        let mut rng = Rng(0x90ab_cdef);
        for _ in 0..4000 {
            let bw = rng.range(200.0, 1000.0);
            let bh = rng.range(200.0, 1000.0);
            let boundary = rect(0.0, 0.0, bw, bh);
            let anchor = rect(
                rng.range(0.0, bw - 20.0),
                rng.range(0.0, bh - 20.0),
                rng.range(4.0, 40.0),
                rng.range(4.0, 40.0),
            );
            let padding = rng.range(0.0, 10.0);
            let side = pick_side(rng.next_u32());

            // Keep the surface small enough that the padded range is valid.
            let floating = if side.is_horizontal() {
                Size::new(rng.range(4.0, 60.0), rng.range(4.0, bh - 2.0 * padding - 1.0))
            } else {
                Size::new(rng.range(4.0, bw - 2.0 * padding - 1.0), rng.range(4.0, 60.0))
            };

            let config = PositionConfig {
                placement: Placement::new(side, pick_align(rng.next_u32())),
                offset: rng.range(0.0, 10.0),
                align_offset: rng.range(-200.0, 200.0),
                flip: false, // isolate the shift step on the requested side
                shift: true,
                padding,
            };
            let placed = position(anchor, floating, boundary, &config);

            if side.is_horizontal() {
                assert!(placed.rect.top() >= boundary.top() + padding - 1e-3);
                assert!(placed.rect.bottom() <= boundary.bottom() - padding + 1e-3);
            } else {
                assert!(placed.rect.left() >= boundary.left() + padding - 1e-3);
                assert!(placed.rect.right() <= boundary.right() - padding + 1e-3);
            }
        }
    }
}
