//! Geometric primitives for input processing.
//!
//! The input system performs hit testing against computed layout rectangles,
//! so it reuses the layout crate's geometry types directly rather than
//! defining parallel ones. This module re-exports [`Point`], [`Size`],
//! [`Rect`], and [`Edges`] from `prism_ui`'s layout crate and adds a few pure
//! helpers used by hit testing and gesture math.
//!
//! All helpers use only basic arithmetic and comparisons; no transcendental
//! functions are used, which keeps results deterministic across platforms.

pub use prism_ui::layout::geometry::{Edges, Point, Rect, Size};

/// Returns `true` when `point` lies inside `rect`.
///
/// The left and top edges are inclusive while the right and bottom edges are
/// exclusive. This half-open convention ensures that two edge-adjacent
/// rectangles never both claim the shared boundary coordinate.
pub fn rect_contains(rect: &Rect, point: Point<f32>) -> bool {
    point.x >= rect.left()
        && point.x < rect.right()
        && point.y >= rect.top()
        && point.y < rect.bottom()
}

/// Returns the squared Euclidean distance between `a` and `b`.
///
/// Squaring avoids a square root, side-stepping the disallowed transcendental
/// functions while remaining monotonic in the true distance. Compare the
/// result against a squared threshold when a radius test is required.
pub fn distance_squared(a: Point<f32>, b: Point<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

/// Returns the absolute value of `value` without using `f32::abs`.
///
/// Provided for callers that prefer an explicit, obviously-pure helper when
/// comparing signed deltas.
pub fn abs_f32(value: f32) -> f32 {
    if value < 0.0 {
        -value
    } else {
        value
    }
}

/// Returns the Manhattan (L1) span between `a` and `b`.
///
/// The L1 metric (`|dx| + |dy|`) is used by the pinch recognizer as a
/// zoom-magnitude proxy because it is monotonic with scale yet avoids the
/// square root required by a true Euclidean distance.
pub fn manhattan_span(a: Point<f32>, b: Point<f32>) -> f32 {
    abs_f32(a.x - b.x) + abs_f32(a.y - b.y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::new(Point::new(x, y), Size::new(w, h))
    }

    #[test]
    fn contains_is_half_open() {
        let r = rect(0.0, 0.0, 10.0, 10.0);
        assert!(rect_contains(&r, Point::new(0.0, 0.0)));
        assert!(rect_contains(&r, Point::new(9.999, 9.999)));
        assert!(!rect_contains(&r, Point::new(10.0, 5.0)));
        assert!(!rect_contains(&r, Point::new(5.0, 10.0)));
        assert!(!rect_contains(&r, Point::new(-0.001, 5.0)));
    }

    #[test]
    fn distance_squared_matches_manual() {
        let d = distance_squared(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
        assert_eq!(d, 25.0);
    }

    #[test]
    fn manhattan_span_is_l1() {
        let s = manhattan_span(Point::new(1.0, 2.0), Point::new(4.0, -2.0));
        assert_eq!(s, 7.0);
        assert_eq!(abs_f32(-3.5), 3.5);
        assert_eq!(abs_f32(2.0), 2.0);
    }
}
