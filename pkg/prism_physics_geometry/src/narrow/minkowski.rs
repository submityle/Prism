//! Shared Minkowski-difference sampling for the GJK/EPA family of queries.
//!
//! Both the boolean/penetration routines (GJK + EPA) and the separated-distance
//! routine evolve a simplex of Minkowski-difference vertices. Each vertex keeps
//! the supporting point it came from on either shape so witness points can be
//! reconstructed by barycentric interpolation. These are clean-room
//! implementations of publicly documented algorithms and contain no Unreal
//! Engine source or derived code.

use glam::Vec3;

use crate::narrow::support::SupportMap;

/// A Minkowski-difference vertex that remembers its supporting points on each
/// shape, enabling per-shape witness-point reconstruction.
#[derive(Clone, Copy)]
pub(crate) struct SupportVertex {
    /// Minkowski-difference position `support_a - support_b`.
    pub v: Vec3,
    /// Supporting point on shape `a`.
    pub a: Vec3,
    /// Supporting point on shape `b`.
    pub b: Vec3,
}

/// Samples the Minkowski difference `A (-) B` along `dir`.
pub(crate) fn support<A: SupportMap, B: SupportMap>(a: &A, b: &B, dir: Vec3) -> SupportVertex {
    let sa = a.support_point(dir);
    let sb = b.support_point(-dir);
    SupportVertex {
        v: sa - sb,
        a: sa,
        b: sb,
    }
}
