//! Contact manifold data produced by the narrow phase and consumed by the
//! contact solver.
//!
//! # Conventions (frozen contract)
//!
//! A [`ContactManifold`] describes the overlap between an ordered pair of
//! bodies `(a, b)`. All narrow-phase generators and all solvers must agree on
//! these conventions:
//!
//! - `normal` is a unit vector pointing **from body `a` toward body `b`**.
//!   Separating the bodies means displacing `a` along `-normal` and `b` along
//!   `+normal`.
//! - Each [`ContactPoint`] stores world-space witness points `point_a` on the
//!   surface of `a` and `point_b` on the surface of `b`, together with a
//!   non-negative `penetration` overlap depth measured along `normal`.
//! - The witness points satisfy `point_a == point_b + penetration * normal`
//!   (up to floating-point error): body `a`'s anchor intrudes toward `b` by
//!   `penetration` along `+normal`.
//!
//! Manifolds carry up to [`MAX_MANIFOLD_POINTS`] points inline, which is enough
//! for the polygonal contact patches produced by box-box clipping and avoids
//! per-contact heap allocation in the hot path.

use crate::state::handle::BodyHandle;
use glam::Vec3;

/// Maximum number of contact points a single manifold can carry.
///
/// Four points is sufficient to represent the quadrilateral contact patch
/// produced by clipping one box face against another, which is the richest
/// primitive-pair manifold in M1.
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// A single contact point within a [`ContactManifold`].
///
/// See the [module documentation](self) for the sign and witness-point
/// conventions.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContactPoint {
    /// World-space witness point on the surface of body `a`.
    pub point_a: Vec3,
    /// World-space witness point on the surface of body `b`.
    pub point_b: Vec3,
    /// Non-negative overlap depth measured along the manifold normal.
    pub penetration: f32,
}

impl ContactPoint {
    /// Creates a contact point from its witness points and penetration depth.
    #[must_use]
    pub fn new(point_a: Vec3, point_b: Vec3, penetration: f32) -> ContactPoint {
        ContactPoint {
            point_a,
            point_b,
            penetration,
        }
    }
}

/// The set of contact points between an ordered pair of bodies.
///
/// The manifold references bodies by [`BodyHandle`]; the narrow phase fills the
/// handles in from the broad-phase pair before returning. Points are stored in
/// a fixed-size inline buffer with an explicit `count` to avoid allocation.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContactManifold {
    /// Handle of the first body in the pair.
    pub body_a: BodyHandle,
    /// Handle of the second body in the pair.
    pub body_b: BodyHandle,
    /// Unit contact normal pointing from body `a` toward body `b`.
    pub normal: Vec3,
    /// Inline contact points; only the first `count` entries are valid.
    points: [ContactPoint; MAX_MANIFOLD_POINTS],
    /// Number of valid entries in `points`.
    count: usize,
}

impl ContactManifold {
    /// Creates an empty manifold with the given bodies and normal.
    ///
    /// The normal should already be unit length and oriented from `a` to `b`.
    #[must_use]
    pub fn new(body_a: BodyHandle, body_b: BodyHandle, normal: Vec3) -> ContactManifold {
        ContactManifold {
            body_a,
            body_b,
            normal,
            points: [ContactPoint::new(Vec3::ZERO, Vec3::ZERO, 0.0); MAX_MANIFOLD_POINTS],
            count: 0,
        }
    }

    /// Appends a contact point, ignoring the request if the manifold is full.
    ///
    /// Returns `true` if the point was stored and `false` if the inline buffer
    /// was already at [`MAX_MANIFOLD_POINTS`].
    pub fn push(&mut self, point: ContactPoint) -> bool {
        if self.count < MAX_MANIFOLD_POINTS {
            self.points[self.count] = point;
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// Returns the valid contact points as a slice.
    #[must_use]
    pub fn points(&self) -> &[ContactPoint] {
        &self.points[..self.count]
    }

    /// Returns the number of valid contact points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the manifold carries no contact points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Sets the body handles on the manifold, returning the modified manifold.
    ///
    /// Narrow-phase generators build manifolds in shape-local terms and then
    /// stamp the owning body handles via this helper.
    #[must_use]
    pub fn with_bodies(mut self, body_a: BodyHandle, body_b: BodyHandle) -> ContactManifold {
        self.body_a = body_a;
        self.body_b = body_b;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_respects_capacity() {
        let mut m = ContactManifold::new(BodyHandle::INVALID, BodyHandle::INVALID, Vec3::Y);
        assert!(m.is_empty());
        for i in 0..MAX_MANIFOLD_POINTS {
            assert!(m.push(ContactPoint::new(Vec3::splat(i as f32), Vec3::ZERO, 0.1)));
        }
        assert_eq!(m.len(), MAX_MANIFOLD_POINTS);
        // Over-capacity push is rejected without panicking.
        assert!(!m.push(ContactPoint::new(Vec3::ZERO, Vec3::ZERO, 0.1)));
        assert_eq!(m.points().len(), MAX_MANIFOLD_POINTS);
    }

    #[test]
    fn witness_point_convention_holds() {
        // Sphere A at origin r=1, sphere B at x=1.5 r=1: overlap 0.5 along +X.
        let normal = Vec3::X;
        let penetration = 0.5;
        let point_a = Vec3::new(1.0, 0.0, 0.0);
        let point_b = Vec3::new(0.5, 0.0, 0.0);
        let reconstructed = point_b + penetration * normal;
        assert!((reconstructed - point_a).length() < 1e-6);
    }
}
