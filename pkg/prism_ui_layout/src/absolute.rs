//! An absolute positioning layout.
//!
//! [`AbsoluteProtocol`] positions each child independently using a CSS-style
//! [`Edges`] inset resolved against the layout bounds. A definite `left`/`top`
//! anchors the child to the start edge; a definite `right`/`bottom` (when the
//! opposite edge is [`Dimension::Auto`]) anchors it to the end edge. Children
//! keep their own measured size, which keeps placement deterministic.

use alloc::vec::Vec;

use crate::geometry::{Dimension, Edges, Point, Rect, Size};
use crate::protocol::{Constraints, LayoutProtocol};

/// A layout that positions children by per-child edge insets.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AbsoluteProtocol {
    /// Per-child insets, matched to children by index. Children without a
    /// corresponding entry are positioned at the bounds' origin.
    pub insets: Vec<Edges<Dimension>>,
}

impl AbsoluteProtocol {
    /// Creates an absolute layout from a list of per-child insets.
    pub fn new(insets: Vec<Edges<Dimension>>) -> Self {
        Self { insets }
    }

    fn inset_for(&self, index: usize) -> Edges<Dimension> {
        self.insets
            .get(index)
            .copied()
            .unwrap_or(Edges::splat(Dimension::Auto))
    }
}

impl LayoutProtocol for AbsoluteProtocol {
    fn measure(&self, _children: &[Size<f32>], constraints: Constraints) -> Size<f32> {
        // Absolutely positioned content fills the space offered to it.
        constraints.max
    }

    fn place(&self, children: &[Size<f32>], bounds: Rect) -> Vec<Rect> {
        let basis_w = Some(bounds.size.width);
        let basis_h = Some(bounds.size.height);
        let mut rects = Vec::with_capacity(children.len());
        for (index, child) in children.iter().enumerate() {
            let inset = self.inset_for(index);
            let x = resolve_axis(
                inset.left.resolve(basis_w),
                inset.right.resolve(basis_w),
                bounds.location.x,
                bounds.size.width,
                child.width,
            );
            let y = resolve_axis(
                inset.top.resolve(basis_h),
                inset.bottom.resolve(basis_h),
                bounds.location.y,
                bounds.size.height,
                child.height,
            );
            rects.push(Rect::new(Point::new(x, y), *child));
        }
        rects
    }
}

/// Resolves a single axis position from optional start/end insets.
fn resolve_axis(start: Option<f32>, end: Option<f32>, origin: f32, extent: f32, child: f32) -> f32 {
    match (start, end) {
        (Some(s), _) => origin + s,
        (None, Some(e)) => origin + extent - e - child,
        (None, None) => origin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_to_start_edges() {
        let proto = AbsoluteProtocol::new(alloc::vec![Edges::new(
            Dimension::Points(10.0),
            Dimension::Auto,
            Dimension::Points(20.0),
            Dimension::Auto,
        )]);
        let children = [Size::new(30.0, 30.0)];
        let bounds = Rect::new(Point::new(5.0, 5.0), Size::new(200.0, 200.0));
        let rects = proto.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(5.0 + 10.0, 5.0 + 20.0));
    }

    #[test]
    fn anchors_to_end_edges() {
        let proto = AbsoluteProtocol::new(alloc::vec![Edges::new(
            Dimension::Auto,
            Dimension::Points(10.0),
            Dimension::Auto,
            Dimension::Points(15.0),
        )]);
        let children = [Size::new(40.0, 20.0)];
        let bounds = Rect::new(Point::ZERO, Size::new(100.0, 100.0));
        let rects = proto.place(&children, bounds);
        // right: 100 - 10 - 40 = 50; bottom: 100 - 15 - 20 = 65.
        assert_eq!(rects[0].location, Point::new(50.0, 65.0));
    }

    #[test]
    fn percent_inset_resolves_against_bounds() {
        let proto = AbsoluteProtocol::new(alloc::vec![Edges::new(
            Dimension::Percent(0.25),
            Dimension::Auto,
            Dimension::Percent(0.5),
            Dimension::Auto,
        )]);
        let children = [Size::new(10.0, 10.0)];
        let bounds = Rect::new(Point::ZERO, Size::new(200.0, 100.0));
        let rects = proto.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(50.0, 50.0));
    }

    #[test]
    fn missing_inset_defaults_to_origin() {
        let proto = AbsoluteProtocol::default();
        let children = [Size::new(10.0, 10.0)];
        let bounds = Rect::new(Point::new(3.0, 4.0), Size::new(50.0, 50.0));
        let rects = proto.place(&children, bounds);
        assert_eq!(rects[0].location, Point::new(3.0, 4.0));
    }

    #[test]
    fn measure_fills_available_space() {
        let proto = AbsoluteProtocol::default();
        let size = proto.measure(&[], Constraints::loose(Size::new(123.0, 45.0)));
        assert_eq!(size, Size::new(123.0, 45.0));
    }
}
