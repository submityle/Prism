//! Multi-point contact manifold: the data a solver needs to keep a resting
//! stack quiet.
//!
//! A single contact point ([`Contact`](super::contact::Contact)) is enough to
//! push two penetrating bodies apart, but it cannot resist *rotation* about the
//! contact: a box resting flat on the ground rocks endlessly if the solver only
//! sees one point. A [`ContactManifold`] carries up to
//! [`MAX_MANIFOLD_POINTS`] coplanar contact points sharing one normal, which is
//! what lets the solver hold a face flush against a face — the behaviour every
//! production engine (`PhysX`, Havok, `Box2D`, Bullet) relies on for stable
//! stacking.
//!
//! # Shape of the data
//!
//! Every point in a manifold shares the same unit `normal` (pointing from body
//! `a` toward body `b`) because the points are, by construction, the corners of
//! one clipped contact polygon lying in a single plane. Each
//! [`ManifoldPoint`] then carries only what differs per corner: its world
//! `position` and its own penetration `depth` along that shared normal. Storing
//! the normal once (rather than per point) matches how a solver iterates a
//! manifold — one normal, several impulses — and keeps the `GPU` record
//! compact.
//!
//! # Fixed capacity
//!
//! Four points fully describe the contact between two convex polyhedra whose
//! deepest features are faces: a quadrilateral-quadrilateral overlap is a
//! convex polygon that four well-chosen corners bound. Capping at four (rather
//! than growing a list) keeps the record a fixed size — essential for the `GPU`
//! port, where each couple writes a constant-stride slot — and matches the
//! four-point cap used by mainstream solvers. The [`count`](ContactManifold)
//! field records how many of the four slots are live.
//!
//! Provenance: textbook contact-manifold representation; no Unreal Engine source
//! or derived code.

use glam::Vec3;

/// Maximum number of contact points a single manifold carries.
///
/// Four corners bound the convex overlap of two box faces, which is the richest
/// contact two oriented boxes can form; deeper feature sets (vertex or edge
/// contacts) use fewer. Fixing the cap keeps every manifold a constant size for
/// the dense `GPU` output layout.
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// One corner of a contact manifold: a world position and its penetration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManifoldPoint {
    /// World-space contact position for this corner.
    pub position: Vec3,
    /// Penetration depth of this corner along the manifold's shared normal;
    /// always positive for a live point.
    pub depth: f32,
}

impl ManifoldPoint {
    /// Assembles a manifold corner from its world position and penetration.
    #[must_use]
    pub fn new(position: Vec3, depth: f32) -> ManifoldPoint {
        ManifoldPoint { position, depth }
    }
}

/// A multi-point contact between two convex bodies.
///
/// The points in [`points`](Self::points) up to [`count`](Self::count) all lie
/// in the contact plane and share the single [`normal`](Self::normal), which
/// runs from body `a` toward body `b` (the direction that separates them).
/// Slots at or beyond `count` are unspecified padding and must not be read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactManifold {
    /// Index of the first body (the `a` side of the contact).
    pub a: u32,
    /// Index of the second body (the `b` side of the contact).
    pub b: u32,
    /// Shared unit contact normal pointing from body `a` toward body `b`.
    pub normal: Vec3,
    /// Number of live points in [`points`](Self::points), `1..=MAX_MANIFOLD_POINTS`.
    pub count: u32,
    /// The contact corners; only the first [`count`](Self::count) are live.
    pub points: [ManifoldPoint; MAX_MANIFOLD_POINTS],
}

impl ContactManifold {
    /// Builds a manifold for `(a, b)` from a shared `normal` and its live
    /// `points`.
    ///
    /// `count` is the number of live entries; slots beyond it are filled with a
    /// zeroed placeholder so the fixed-size array is always fully initialised.
    ///
    /// # Panics
    ///
    /// Panics if `count` is zero or exceeds [`MAX_MANIFOLD_POINTS`], or if fewer
    /// than `count` points are supplied, since a reported manifold always has at
    /// least one and at most four live corners.
    #[must_use]
    pub fn new(
        a: u32,
        b: u32,
        normal: Vec3,
        count: usize,
        points: &[ManifoldPoint],
    ) -> ContactManifold {
        assert!(
            (1..=MAX_MANIFOLD_POINTS).contains(&count),
            "a manifold carries one to four points"
        );
        assert!(
            points.len() >= count,
            "fewer points supplied than the live count"
        );
        let placeholder = ManifoldPoint::new(Vec3::ZERO, 0.0);
        let mut slots = [placeholder; MAX_MANIFOLD_POINTS];
        for (slot, point) in slots.iter_mut().zip(points.iter()).take(count) {
            *slot = *point;
        }
        ContactManifold {
            a,
            b,
            normal,
            count: count as u32,
            points: slots,
        }
    }
}
