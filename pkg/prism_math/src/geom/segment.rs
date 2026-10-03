//! Finite line segment [`Segment3`].

use crate::vec::Vec3;

/// A finite line segment between two endpoints `start` and `end`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Segment3 {
    /// The start endpoint (parameter `t = 0`).
    pub start: Vec3,
    /// The end endpoint (parameter `t = 1`).
    pub end: Vec3,
}

impl Segment3 {
    /// Create a segment from two endpoints.
    #[inline]
    pub const fn new(start: Vec3, end: Vec3) -> Self {
        Self { start, end }
    }

    /// The direction from `start` to `end` (not normalized).
    #[inline]
    pub fn direction(self) -> Vec3 {
        self.end - self.start
    }

    /// The segment length.
    #[inline]
    pub fn length(self) -> f32 {
        self.direction().length()
    }

    /// The squared segment length.
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.direction().length_squared()
    }

    /// Evaluate the point at parameter `t`, where `t = 0` is `start` and
    /// `t = 1` is `end`. Values outside `[0, 1]` extrapolate.
    #[inline]
    pub fn at(self, t: f32) -> Vec3 {
        self.start + self.direction() * t
    }

    /// The parameter `t` in `[0, 1]` of the point on the segment closest to
    /// `p`. Returns `0` for a degenerate (zero-length) segment.
    #[inline]
    pub fn closest_t(self, p: Vec3) -> f32 {
        let d = self.direction();
        let len2 = d.length_squared();
        if len2 <= 1.0e-20 {
            return 0.0;
        }
        ((p - self.start).dot(d) / len2).clamp(0.0, 1.0)
    }

    /// The point on the segment closest to `p`.
    #[inline]
    pub fn closest_point(self, p: Vec3) -> Vec3 {
        self.at(self.closest_t(p))
    }

    /// Squared distance from `p` to the segment.
    #[inline]
    pub fn distance_squared(self, p: Vec3) -> f32 {
        (p - self.closest_point(p)).length_squared()
    }
}
