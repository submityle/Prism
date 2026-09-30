//! A generated sphere-sphere contact.
//!
//! A [`Contact`] is the narrow phase's output for one colliding pair: the two
//! particle indices, the unit contact `normal` pointing from `a` to `b`, the
//! positive penetration `depth` (how far the spheres overlap along the normal),
//! and the world-space contact `point` placed on the mid-overlap plane. This is
//! exactly the manifold a position-based or impulse solver consumes: push the
//! two bodies apart by `depth` along `normal`, applying the correction at
//! `point`.
//!
//! The narrow phase emits one slot per input candidate pair, so a pair that
//! turns out not to penetrate yields [`None`] in that slot rather than being
//! dropped; compacting the survivors is a separate scan stage. This keeps the
//! contact index aligned with the pair index, which the parity test relies on.
//!
//! Provenance: textbook sphere-sphere manifold; no Unreal Engine source or
//! derived code.

use glam::Vec3;

/// A contact between two overlapping bounding spheres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Index of the first particle (the `a` side of the pair).
    pub a: u32,
    /// Index of the second particle (the `b` side of the pair).
    pub b: u32,
    /// Unit contact normal pointing from particle `a` toward particle `b`.
    pub normal: Vec3,
    /// Penetration depth along [`normal`](Self::normal); always positive for a
    /// reported contact.
    pub depth: f32,
    /// World-space contact point on the plane midway through the overlap.
    pub point: Vec3,
}

impl Contact {
    /// Assembles a contact for the pair `(a, b)` from its geometry.
    #[must_use]
    pub fn new(a: u32, b: u32, normal: Vec3, depth: f32, point: Vec3) -> Contact {
        Contact {
            a,
            b,
            normal,
            depth,
            point,
        }
    }
}
