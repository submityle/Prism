//! A wrapping flow layout.
//!
//! [`WrapProtocol`] places children left-to-right along the main axis,
//! breaking onto a new line whenever the next child would exceed the main
//! extent. Each line is as tall as its tallest child, and lines stack along
//! the cross axis separated by the cross gap. This mirrors CSS flex-wrap and
//! Compose's `FlowRow`.

use alloc::vec::Vec;

use crate::geometry::{Point, Rect, Size};
use crate::protocol::{Constraints, LayoutProtocol};

/// A layout that flows children into rows, wrapping on overflow.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct WrapProtocol {
    /// Gap between adjacent children on the main (horizontal) axis.
    pub main_gap: f32,
    /// Gap between adjacent lines on the cross (vertical) axis.
    pub cross_gap: f32,
}

impl WrapProtocol {
    /// Creates a wrap layout with the given main- and cross-axis gaps.
    pub fn new(main_gap: f32, cross_gap: f32) -> Self {
        Self {
            main_gap,
            cross_gap,
        }
    }

    /// Groups child indices into lines that each fit within `max_main`.
    ///
    /// A line always contains at least one child, even when that single child
    /// is wider than `max_main`, to guarantee termination.
    fn lines(&self, children: &[Size<f32>], max_main: f32) -> Vec<Line> {
        let mut lines = Vec::new();
        let mut current = Line::default();
        for (index, child) in children.iter().enumerate() {
            let gap = if current.count == 0 {
                0.0
            } else {
                self.main_gap
            };
            let projected = current.main + gap + child.width;
            if current.count > 0 && projected > max_main {
                lines.push(current);
                current = Line::default();
                current.start = index;
            }
            if current.count == 0 {
                current.start = index;
                current.main = child.width;
            } else {
                current.main += gap + child.width;
            }
            current.count += 1;
            current.cross = current.cross.max(child.height);
        }
        if current.count > 0 {
            lines.push(current);
        }
        lines
    }
}

/// A single flowed line: a contiguous run of children plus its extents.
#[derive(Clone, Copy, Debug, Default)]
struct Line {
    start: usize,
    count: usize,
    main: f32,
    cross: f32,
}

impl LayoutProtocol for WrapProtocol {
    fn measure(&self, children: &[Size<f32>], constraints: Constraints) -> Size<f32> {
        let lines = self.lines(children, constraints.max.width);
        let mut width = 0.0_f32;
        let mut height = 0.0_f32;
        for (line_index, line) in lines.iter().enumerate() {
            width = width.max(line.main);
            if line_index > 0 {
                height += self.cross_gap;
            }
            height += line.cross;
        }
        constraints.constrain(Size::new(width, height))
    }

    fn place(&self, children: &[Size<f32>], bounds: Rect) -> Vec<Rect> {
        let lines = self.lines(children, bounds.size.width);
        let mut rects = alloc::vec![Rect::new(Point::ZERO, Size::ZERO); children.len()];
        let mut cursor_y = bounds.location.y;
        for line in &lines {
            let mut cursor_x = bounds.location.x;
            for offset in 0..line.count {
                let index = line.start + offset;
                let child = children[index];
                rects[index] = Rect::new(Point::new(cursor_x, cursor_y), child);
                cursor_x += child.width + self.main_gap;
            }
            cursor_y += line.cross + self.cross_gap;
        }
        rects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_when_main_axis_overflows() {
        let wrap = WrapProtocol::new(10.0, 5.0);
        let children = [
            Size::new(40.0, 20.0),
            Size::new(40.0, 20.0),
            Size::new(40.0, 30.0),
        ];
        // Only two 40-wide children (+10 gap = 90) fit in 100; third wraps.
        let bounds = Rect::new(Point::ZERO, Size::new(100.0, 200.0));
        let rects = wrap.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(0.0, 0.0));
        assert_eq!(rects[1].location, Point::new(50.0, 0.0));
        // New line starts below the first line's height (20) + cross gap (5).
        assert_eq!(rects[2].location, Point::new(0.0, 25.0));
    }

    #[test]
    fn single_line_when_everything_fits() {
        let wrap = WrapProtocol::new(0.0, 0.0);
        let children = [Size::new(20.0, 10.0), Size::new(20.0, 10.0)];
        let bounds = Rect::new(Point::ZERO, Size::new(100.0, 100.0));
        let rects = wrap.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(0.0, 0.0));
        assert_eq!(rects[1].location, Point::new(20.0, 0.0));
    }

    #[test]
    fn oversized_child_occupies_its_own_line() {
        let wrap = WrapProtocol::new(0.0, 0.0);
        let children = [Size::new(200.0, 10.0), Size::new(10.0, 10.0)];
        let bounds = Rect::new(Point::ZERO, Size::new(100.0, 100.0));
        let rects = wrap.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(0.0, 0.0));
        assert_eq!(rects[1].location, Point::new(0.0, 10.0));
    }

    #[test]
    fn measure_reports_widest_line_and_stacked_height() {
        let wrap = WrapProtocol::new(10.0, 5.0);
        let children = [
            Size::new(40.0, 20.0),
            Size::new(40.0, 20.0),
            Size::new(40.0, 30.0),
        ];
        let size = wrap.measure(&children, Constraints::loose(Size::new(100.0, 500.0)));
        // Line 1 width = 90, line 2 width = 40 -> 90; height = 20 + 5 + 30.
        assert_eq!(size, Size::new(90.0, 55.0));
    }
}
