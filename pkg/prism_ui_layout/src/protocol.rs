//! The composable layout protocol.
//!
//! [`LayoutProtocol`] is the extension point that lets custom layouts become
//! first-class citizens alongside the built-in flex solver, mirroring
//! `SwiftUI`'s `Layout` protocol and Flutter's `RenderObject`. A protocol is a
//! pure function of its children's measured sizes and an incoming
//! [`Constraints`]: it reports a desired [`Size`] from
//! [`LayoutProtocol::measure`] and a deterministic set of child rectangles
//! from [`LayoutProtocol::place`].
//!
//! Because both methods are pure, every built-in protocol
//! ([`FlexProtocol`], [`crate::grid::GridProtocol`],
//! [`crate::stack::StackProtocol`], [`crate::absolute::AbsoluteProtocol`],
//! [`crate::wrap::WrapProtocol`]) is directly unit-testable: supply child
//! sizes and bounds, then assert on the placed rectangles.

use alloc::vec::Vec;

use crate::geometry::{Point, Rect, Size};
use crate::style::{AlignItems, FlexDirection, JustifyContent};

/// Box constraints: a closed range of permitted sizes on each axis.
///
/// A protocol must return a size within `[min, max]` on both axes. Tight
/// constraints (`min == max`) force an exact size; loose constraints
/// (`min == 0`) allow any size up to `max`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constraints {
    /// Smallest permitted size on each axis.
    pub min: Size<f32>,
    /// Largest permitted size on each axis.
    pub max: Size<f32>,
}

impl Constraints {
    /// Creates constraints from an explicit `min` and `max`.
    pub const fn new(min: Size<f32>, max: Size<f32>) -> Self {
        Self { min, max }
    }

    /// Creates tight constraints that force exactly `size`.
    pub const fn tight(size: Size<f32>) -> Self {
        Self {
            min: size,
            max: size,
        }
    }

    /// Creates loose constraints: any size from zero up to `max`.
    pub const fn loose(max: Size<f32>) -> Self {
        Self {
            min: Size::ZERO,
            max,
        }
    }

    /// Clamps `size` into the permitted range on both axes.
    pub fn constrain(&self, size: Size<f32>) -> Size<f32> {
        Size::new(
            size.width.clamp(self.min.width, self.max.width),
            size.height.clamp(self.min.height, self.max.height),
        )
    }
}

/// Horizontal alignment of a smaller box inside a larger one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlignX {
    /// Align to the left edge.
    #[default]
    Start,
    /// Center horizontally.
    Center,
    /// Align to the right edge.
    End,
}

/// Vertical alignment of a smaller box inside a larger one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlignY {
    /// Align to the top edge.
    #[default]
    Start,
    /// Center vertically.
    Center,
    /// Align to the bottom edge.
    End,
}

/// A 2D alignment combining a horizontal and vertical anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Alignment {
    /// Horizontal anchor.
    pub x: AlignX,
    /// Vertical anchor.
    pub y: AlignY,
}

impl Alignment {
    /// Top-left alignment.
    pub const TOP_LEFT: Self = Self {
        x: AlignX::Start,
        y: AlignY::Start,
    };

    /// Center alignment on both axes.
    pub const CENTER: Self = Self {
        x: AlignX::Center,
        y: AlignY::Center,
    };

    /// Bottom-right alignment.
    pub const BOTTOM_RIGHT: Self = Self {
        x: AlignX::End,
        y: AlignY::End,
    };

    /// Creates an alignment from its horizontal and vertical anchors.
    pub const fn new(x: AlignX, y: AlignY) -> Self {
        Self { x, y }
    }

    /// Returns the top-left offset that positions an `inner` box inside an
    /// `outer` box according to this alignment.
    ///
    /// Overflow (an `inner` larger than `outer`) is permitted and produces a
    /// negative offset for centered or end anchors, matching how overflowing
    /// content hangs outside its container.
    pub fn offset(self, outer: Size<f32>, inner: Size<f32>) -> Point<f32> {
        let free_x = outer.width - inner.width;
        let free_y = outer.height - inner.height;
        let x = match self.x {
            AlignX::Start => 0.0,
            AlignX::Center => free_x / 2.0,
            AlignX::End => free_x,
        };
        let y = match self.y {
            AlignY::Start => 0.0,
            AlignY::Center => free_y / 2.0,
            AlignY::End => free_y,
        };
        Point::new(x, y)
    }
}

/// A composable layout algorithm.
///
/// Implementors are pure: given the children's measured sizes and the
/// incoming constraints they must deterministically report a desired size and
/// a child-rectangle set. The rectangles returned by
/// [`LayoutProtocol::place`] are expressed relative to `bounds`' top-left
/// corner translated into absolute coordinates (that is, they already include
/// `bounds.location`).
pub trait LayoutProtocol {
    /// Reports the size this layout wants, given its children and the
    /// incoming constraints. The result is always within `constraints`.
    fn measure(&self, children: &[Size<f32>], constraints: Constraints) -> Size<f32>;

    /// Produces one rectangle per child, positioned within `bounds`.
    ///
    /// The returned vector has the same length and order as `children`.
    fn place(&self, children: &[Size<f32>], bounds: Rect) -> Vec<Rect>;
}

/// A single-line flex layout exposed through [`LayoutProtocol`].
///
/// This adapter provides the composable-protocol surface for flex-style line
/// layout (main-axis stacking with a gap, main-axis distribution, and
/// cross-axis alignment). The full tree solver remains in
/// [`crate::tree::LayoutTree::compute_layout`]; this type operates on an
/// already-measured slice of child sizes so it can be unit-tested in
/// isolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlexProtocol {
    /// Direction of the main axis.
    pub direction: FlexDirection,
    /// Gap inserted between adjacent children on the main axis.
    pub gap: f32,
    /// Main-axis distribution of children.
    pub justify: JustifyContent,
    /// Cross-axis alignment of children.
    pub align: AlignItems,
}

impl Default for FlexProtocol {
    fn default() -> Self {
        Self {
            direction: FlexDirection::Row,
            gap: 0.0,
            justify: JustifyContent::Start,
            align: AlignItems::Start,
        }
    }
}

impl FlexProtocol {
    /// Creates a flex protocol with the given direction, leaving other
    /// fields at their defaults.
    pub fn new(direction: FlexDirection) -> Self {
        Self {
            direction,
            ..Self::default()
        }
    }

    fn content_main(&self, children: &[Size<f32>]) -> f32 {
        let is_row = self.direction.is_row();
        let mut total = 0.0;
        for child in children {
            total += child.main(is_row);
        }
        total + self.gap * gaps(children.len())
    }

    fn content_cross(&self, children: &[Size<f32>]) -> f32 {
        let is_row = self.direction.is_row();
        let mut max = 0.0_f32;
        for child in children {
            max = max.max(child.cross(is_row));
        }
        max
    }
}

impl LayoutProtocol for FlexProtocol {
    fn measure(&self, children: &[Size<f32>], constraints: Constraints) -> Size<f32> {
        let is_row = self.direction.is_row();
        let main = self.content_main(children);
        let cross = self.content_cross(children);
        let desired = if is_row {
            Size::new(main, cross)
        } else {
            Size::new(cross, main)
        };
        constraints.constrain(desired)
    }

    fn place(&self, children: &[Size<f32>], bounds: Rect) -> Vec<Rect> {
        let is_row = self.direction.is_row();
        let reverse = self.direction.is_reverse();
        let main_extent = bounds.size.main(is_row);
        let cross_extent = bounds.size.cross(is_row);

        let content_main = self.content_main(children);
        let free = main_extent - content_main;
        let (lead, between) = justify(self.justify, free, children.len(), self.gap);

        let mut cursor = lead;
        let mut rects = Vec::with_capacity(children.len());
        for child in children {
            let child_main = child.main(is_row);
            let child_cross = child.cross(is_row);
            let cross_free = cross_extent - child_cross;
            let cross_offset = match self.align {
                AlignItems::Start | AlignItems::Stretch => 0.0,
                AlignItems::Center => cross_free / 2.0,
                AlignItems::End => cross_free,
            };
            let main_pos = if reverse {
                main_extent - cursor - child_main
            } else {
                cursor
            };
            let location = if is_row {
                Point::new(
                    bounds.location.x + main_pos,
                    bounds.location.y + cross_offset,
                )
            } else {
                Point::new(
                    bounds.location.x + cross_offset,
                    bounds.location.y + main_pos,
                )
            };
            rects.push(Rect::new(location, *child));
            cursor += child_main + between;
        }
        rects
    }
}

/// Returns the number of gaps between `count` items (zero when fewer than
/// two).
pub(crate) fn gaps(count: usize) -> f32 {
    if count <= 1 {
        0.0
    } else {
        (count - 1) as f32
    }
}

/// Computes the leading offset and inter-item spacing for a justify mode.
fn justify(mode: JustifyContent, free: f32, count: usize, gap: f32) -> (f32, f32) {
    let n = count as f32;
    match mode {
        JustifyContent::Start => (0.0, gap),
        JustifyContent::End => (free, gap),
        JustifyContent::Center => (free / 2.0, gap),
        JustifyContent::SpaceBetween => {
            if count <= 1 {
                (0.0, gap)
            } else {
                (0.0, gap + free / (n - 1.0))
            }
        }
        JustifyContent::SpaceAround => {
            if count == 0 {
                (0.0, gap)
            } else {
                let unit = free / n;
                (unit / 2.0, gap + unit)
            }
        }
        JustifyContent::SpaceEvenly => {
            let unit = free / (n + 1.0);
            (unit, gap + unit)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrain_clamps_both_axes() {
        let c = Constraints::new(Size::new(10.0, 10.0), Size::new(100.0, 50.0));
        assert_eq!(c.constrain(Size::new(5.0, 200.0)), Size::new(10.0, 50.0));
        assert_eq!(c.constrain(Size::new(40.0, 25.0)), Size::new(40.0, 25.0));
    }

    #[test]
    fn tight_and_loose_constructors() {
        let t = Constraints::tight(Size::new(20.0, 30.0));
        assert_eq!(t.min, t.max);
        let l = Constraints::loose(Size::new(20.0, 30.0));
        assert_eq!(l.min, Size::ZERO);
        assert_eq!(l.max, Size::new(20.0, 30.0));
    }

    #[test]
    fn alignment_offset_center_and_end() {
        let outer = Size::new(100.0, 100.0);
        let inner = Size::new(40.0, 20.0);
        assert_eq!(
            Alignment::CENTER.offset(outer, inner),
            Point::new(30.0, 40.0)
        );
        assert_eq!(
            Alignment::BOTTOM_RIGHT.offset(outer, inner),
            Point::new(60.0, 80.0)
        );
        assert_eq!(
            Alignment::TOP_LEFT.offset(outer, inner),
            Point::new(0.0, 0.0)
        );
    }

    #[test]
    fn flex_row_places_children_with_gap() {
        let proto = FlexProtocol {
            direction: FlexDirection::Row,
            gap: 10.0,
            justify: JustifyContent::Start,
            align: AlignItems::Start,
        };
        let children = [Size::new(20.0, 20.0), Size::new(30.0, 40.0)];
        let bounds = Rect::new(Point::new(5.0, 5.0), Size::new(300.0, 100.0));
        let rects = proto.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(5.0, 5.0));
        assert_eq!(rects[1].location, Point::new(5.0 + 20.0 + 10.0, 5.0));
    }

    #[test]
    fn flex_row_space_between() {
        let proto = FlexProtocol {
            direction: FlexDirection::Row,
            gap: 0.0,
            justify: JustifyContent::SpaceBetween,
            align: AlignItems::Start,
        };
        let children = [Size::new(50.0, 10.0), Size::new(50.0, 10.0)];
        let bounds = Rect::new(Point::ZERO, Size::new(300.0, 100.0));
        let rects = proto.place(&children, bounds);
        assert_eq!(rects[0].location.x, 0.0);
        assert_eq!(rects[1].location.x, 250.0);
    }

    #[test]
    fn flex_column_center_cross_alignment() {
        let proto = FlexProtocol {
            direction: FlexDirection::Column,
            gap: 0.0,
            justify: JustifyContent::Start,
            align: AlignItems::Center,
        };
        let children = [Size::new(40.0, 20.0)];
        let bounds = Rect::new(Point::ZERO, Size::new(100.0, 100.0));
        let rects = proto.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(30.0, 0.0));
    }

    #[test]
    fn flex_measure_respects_constraints() {
        let proto = FlexProtocol::new(FlexDirection::Row);
        let children = [Size::new(50.0, 10.0), Size::new(50.0, 30.0)];
        let size = proto.measure(&children, Constraints::loose(Size::new(500.0, 500.0)));
        assert_eq!(size, Size::new(100.0, 30.0));
    }

    #[test]
    fn flex_row_reverse_places_from_end() {
        let proto = FlexProtocol {
            direction: FlexDirection::RowReverse,
            gap: 0.0,
            justify: JustifyContent::Start,
            align: AlignItems::Start,
        };
        let children = [Size::new(20.0, 10.0), Size::new(30.0, 10.0)];
        let bounds = Rect::new(Point::ZERO, Size::new(100.0, 50.0));
        let rects = proto.place(&children, bounds);
        // First child sits flush against the right edge.
        assert_eq!(rects[0].location.x, 80.0);
        assert_eq!(rects[1].location.x, 50.0);
    }
}
