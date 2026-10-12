//! Axis-aligned bounding boxes over point sets and rects.
//!
//! Culling, layout, and spatial hashing all want the tight axis-aligned box
//! around a run of points, plus the handful of box operations — union, expand by
//! a margin, containment — that go with it. [`bounds`] computes the tight box;
//! [`union`], [`expand`] and [`contains`] are the operations. Everything is
//! expressed with the layout [`Rect`] so a result drops straight into hit-testing
//! or culling.
//!
//! Only `+ - *` and comparisons on `f32` are used, so the module is
//! `no_std`-clean and bit-stable.

use prism_ui_layout::{Point, Rect, Size};

/// The tight axis-aligned bounding box of `points`, or `None` when empty.
///
/// The box's four sides each touch at least one input point. A single point
/// gives a zero-size rect at that point. `NaN` coordinates are ignored by
/// [`f32::min`]/[`f32::max`] rather than poisoning the box.
#[must_use]
pub fn bounds(points: &[(f32, f32)]) -> Option<Rect> {
    let mut iter = points.iter();
    let &(mut min_x, mut min_y) = iter.next()?;
    let (mut max_x, mut max_y) = (min_x, min_y);
    for &(x, y) in iter {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    Some(Rect::new(
        Point::new(min_x, min_y),
        Size::new(max_x - min_x, max_y - min_y),
    ))
}

/// The smallest rect containing both `a` and `b`.
#[must_use]
pub fn union(a: Rect, b: Rect) -> Rect {
    let min_x = a.left().min(b.left());
    let min_y = a.top().min(b.top());
    let max_x = a.right().max(b.right());
    let max_y = a.bottom().max(b.bottom());
    Rect::new(
        Point::new(min_x, min_y),
        Size::new(max_x - min_x, max_y - min_y),
    )
}

/// Grow `rect` by `margin` on every side.
///
/// A negative `margin` shrinks the box; each extent is clamped at zero, so the
/// result never has a negative width or height.
#[must_use]
pub fn expand(rect: Rect, margin: f32) -> Rect {
    let width = (rect.size.width + 2.0 * margin).max(0.0);
    let height = (rect.size.height + 2.0 * margin).max(0.0);
    Rect::new(
        Point::new(rect.left() - margin, rect.top() - margin),
        Size::new(width, height),
    )
}

/// Whether `p` lies within `rect` (all four edges inclusive).
#[must_use]
pub fn contains(rect: Rect, p: (f32, f32)) -> bool {
    p.0 >= rect.left() && p.0 <= rect.right() && p.1 >= rect.top() && p.1 <= rect.bottom()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::new(Point::new(x, y), Size::new(w, h))
    }

    #[test]
    fn bounds_is_tight_and_contains_every_point() {
        let pts = [(3.0, -1.0), (-2.0, 5.0), (7.0, 2.0), (0.0, 0.0)];
        let b = bounds(&pts).expect("non-empty");
        assert_eq!(b, rect(-2.0, -1.0, 9.0, 6.0));
        for &p in &pts {
            assert!(contains(b, p));
        }
        assert_eq!(b.left(), -2.0);
        assert_eq!(b.right(), 7.0);
        assert_eq!(b.top(), -1.0);
        assert_eq!(b.bottom(), 5.0);
    }

    #[test]
    fn bounds_of_empty_and_single_point() {
        assert!(bounds(&[]).is_none());
        assert_eq!(bounds(&[(4.0, -3.0)]), Some(rect(4.0, -3.0, 0.0, 0.0)));
    }

    #[test]
    fn union_is_the_smallest_covering_rect() {
        let u = union(rect(0.0, 0.0, 2.0, 2.0), rect(5.0, -1.0, 3.0, 4.0));
        assert_eq!(u, rect(0.0, -1.0, 8.0, 4.0));
        assert!(contains(u, (0.0, 0.0)) && contains(u, (8.0, 3.0)));
    }

    #[test]
    fn expand_grows_and_never_goes_negative() {
        let r = rect(1.0, 2.0, 4.0, 6.0);
        assert_eq!(expand(r, 2.0), rect(-1.0, 0.0, 8.0, 10.0));
        let tiny = expand(r, -10.0);
        assert_eq!(tiny.size.width, 0.0);
        assert_eq!(tiny.size.height, 0.0);
    }

    #[test]
    fn contains_is_edge_inclusive() {
        let r = rect(0.0, 0.0, 4.0, 4.0);
        assert!(contains(r, (0.0, 0.0)) && contains(r, (4.0, 4.0)));
        assert!(contains(r, (2.0, 2.0)));
        assert!(!contains(r, (-0.1, 2.0)) && !contains(r, (2.0, 4.1)));
    }
}
