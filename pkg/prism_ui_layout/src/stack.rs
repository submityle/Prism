//! A z-stacking layout.
//!
//! [`StackProtocol`] overlaps all of its children in the same region,
//! painting them in declaration order (first child at the back). Every child
//! is aligned within the stack's bounds using a shared [`Alignment`], which
//! makes it the building block for badges, overlays, and centered content.

use alloc::vec::Vec;

use crate::geometry::{Point, Rect, Size};
use crate::protocol::{Alignment, Constraints, LayoutProtocol};

/// A layout that overlaps its children, aligning each within the bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct StackProtocol {
    /// Alignment applied to every child within the stack bounds.
    pub alignment: Alignment,
}

impl StackProtocol {
    /// Creates a stack with the given alignment.
    pub fn new(alignment: Alignment) -> Self {
        Self { alignment }
    }
}

impl LayoutProtocol for StackProtocol {
    fn measure(&self, children: &[Size<f32>], constraints: Constraints) -> Size<f32> {
        let mut width = 0.0_f32;
        let mut height = 0.0_f32;
        for child in children {
            width = width.max(child.width);
            height = height.max(child.height);
        }
        constraints.constrain(Size::new(width, height))
    }

    fn place(&self, children: &[Size<f32>], bounds: Rect) -> Vec<Rect> {
        let mut rects = Vec::with_capacity(children.len());
        for child in children {
            let offset = self.alignment.offset(bounds.size, *child);
            let location = Point::new(bounds.location.x + offset.x, bounds.location.y + offset.y);
            rects.push(Rect::new(location, *child));
        }
        rects
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{AlignX, AlignY};

    #[test]
    fn stack_centers_every_child() {
        let stack = StackProtocol::new(Alignment::CENTER);
        let children = [Size::new(40.0, 40.0), Size::new(20.0, 20.0)];
        let bounds = Rect::new(Point::ZERO, Size::new(100.0, 100.0));
        let rects = stack.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(30.0, 30.0));
        assert_eq!(rects[1].location, Point::new(40.0, 40.0));
    }

    #[test]
    fn stack_measure_is_max_of_children() {
        let stack = StackProtocol::default();
        let children = [Size::new(40.0, 10.0), Size::new(20.0, 50.0)];
        let size = stack.measure(&children, Constraints::loose(Size::new(500.0, 500.0)));
        assert_eq!(size, Size::new(40.0, 50.0));
    }

    #[test]
    fn stack_respects_corner_alignment() {
        let stack = StackProtocol::new(Alignment::new(AlignX::End, AlignY::Start));
        let children = [Size::new(20.0, 20.0)];
        let bounds = Rect::new(Point::new(5.0, 5.0), Size::new(100.0, 100.0));
        let rects = stack.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(5.0 + 80.0, 5.0));
    }

    #[test]
    fn stack_offsets_relative_to_bounds_origin() {
        let stack = StackProtocol::new(Alignment::TOP_LEFT);
        let children = [Size::new(10.0, 10.0)];
        let bounds = Rect::new(Point::new(7.0, 9.0), Size::new(50.0, 50.0));
        let rects = stack.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(7.0, 9.0));
    }
}
