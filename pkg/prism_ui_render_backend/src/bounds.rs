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

/// The overlap of `a` and `b`, or `None` when they are disjoint.
///
/// A pair that touches only along an edge or at a corner yields a `Some`
/// zero-size rect (the shared boundary), matching the edge-inclusive
/// [`intersects`](crate::draw::intersects) predicate.
#[must_use]
pub fn intersection(a: Rect, b: Rect) -> Option<Rect> {
    let x0 = a.left().max(b.left());
    let x1 = a.right().min(b.right());
    let y0 = a.top().max(b.top());
    let y1 = a.bottom().min(b.bottom());
    if x0 > x1 || y0 > y1 {
        return None;
    }
    Some(Rect::new(
        Point::new(x0, y0),
        Size::new(x1 - x0, y1 - y0),
    ))
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

/// The intersection of every rect in `stack`, or `None` when `stack` is empty.
///
/// Used to fold a stack of enclosing layer bounds into the single active
/// scissor rect that both the CPU reference rasteriser and the GPU compute
/// shader apply per primitive. `None` means "unbounded"; a nested pair that does
/// not overlap yields a rect with a negative extent, which [`contains`] rejects
/// for every point — i.e. an empty clip clips everything away.
#[must_use]
pub fn clip_of(stack: &[Rect]) -> Option<Rect> {
    let (&first, rest) = stack.split_first()?;
    let mut acc = first;
    for &r in rest {
        let x0 = acc.left().max(r.left());
        let x1 = acc.right().min(r.right());
        let y0 = acc.top().max(r.top());
        let y1 = acc.bottom().min(r.bottom());
        acc = Rect::new(Point::new(x0, y0), Size::new(x1 - x0, y1 - y0));
    }
    Some(acc)
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

    #[test]
    fn intersection_is_the_overlap_or_none() {
        let a = rect(0.0, 0.0, 10.0, 10.0);
        assert_eq!(
            intersection(a, rect(5.0, 5.0, 10.0, 10.0)),
            Some(rect(5.0, 5.0, 5.0, 5.0))
        );
        // A contained rect is returned whole.
        assert_eq!(
            intersection(a, rect(2.0, 2.0, 3.0, 3.0)),
            Some(rect(2.0, 2.0, 3.0, 3.0))
        );
        // Disjoint.
        assert!(intersection(a, rect(20.0, 20.0, 1.0, 1.0)).is_none());
        // Edge touch -> a zero-size rect at the shared edge.
        assert_eq!(
            intersection(a, rect(10.0, 0.0, 5.0, 5.0)),
            Some(rect(10.0, 0.0, 0.0, 5.0))
        );
        assert_eq!(intersection(a, a), Some(a));
    }

    #[test]
    fn intersection_agrees_with_the_intersects_predicate() {
        let xs = [-5.0_f32, -2.0, 0.0, 2.0, 4.0, 5.0, 6.0, 9.0];
        for &ax in &xs {
            for &bx in &xs {
                let a = rect(ax, 0.0, 5.0, 5.0);
                let b = rect(bx, 2.0, 5.0, 5.0);
                assert_eq!(
                    intersection(a, b).is_some(),
                    crate::draw::intersects(a, b),
                    "disagree for {a:?} {b:?}"
                );
            }
        }
    }

    #[test]
    fn clip_of_is_empty_for_no_layers_and_folds_nested_bounds() {
        assert_eq!(clip_of(&[]), None);
        // A single layer clips to exactly its bounds.
        assert_eq!(clip_of(&[rect(1.0, 2.0, 5.0, 6.0)]), Some(rect(1.0, 2.0, 5.0, 6.0)));
        // Nested bounds intersect.
        let nested = [rect(0.0, 0.0, 10.0, 10.0), rect(4.0, 4.0, 20.0, 20.0)];
        assert_eq!(clip_of(&nested), Some(rect(4.0, 4.0, 6.0, 6.0)));
        // Disjoint nested bounds clip everything away (negative extent).
        let disjoint = [rect(0.0, 0.0, 2.0, 2.0), rect(5.0, 5.0, 2.0, 2.0)];
        let empty = clip_of(&disjoint).expect("non-empty stack");
        assert!(empty.size.width < 0.0 && empty.size.height < 0.0);
        assert!(!contains(empty, (1.0, 1.0)) && !contains(empty, (6.0, 6.0)));
    }
}
