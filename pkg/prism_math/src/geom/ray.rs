//! Parametric 3D ray [`Ray3`].

use crate::vec::Vec3;

/// A half-line defined by an `origin` and a `direction`.
///
/// Points on the ray are `origin + direction * t` for `t >= 0`. The direction
/// is not required to be unit length by the constructor, but intersection
/// routines document whether they assume a normalized direction.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Ray3 {
    /// The ray origin.
    pub origin: Vec3,
    /// The ray direction (not necessarily unit length).
    pub direction: Vec3,
}

impl Ray3 {
    /// Create a ray from an `origin` and a `direction`.
    #[inline]
    pub const fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// Create a ray whose `direction` is normalized to unit length.
    #[inline]
    pub fn new_normalized(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalize() }
    }

    /// Evaluate the point at parameter `t`: `origin + direction * t`.
    #[inline]
    pub fn at(self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Return a copy with a unit-length `direction`.
    #[inline]
    pub fn normalized(self) -> Self {
        Self { origin: self.origin, direction: self.direction.normalize() }
    }
}
