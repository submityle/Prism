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

/// Reduces a set of coplanar contact points to at most
/// [`MAX_MANIFOLD_POINTS`], keeping the widest, deepest quad.
///
/// Picks the deepest corner, the corner farthest from it, then the two corners
/// that maximise the signed triangle area to either side of that diagonal
/// (measured in the plane whose normal is `normal`). Four or fewer input
/// points are returned unchanged, preserving order. This is the standard
/// four-point manifold reduction every mesh and polyhedron collider in this
/// crate shares, so the kept quad cannot drift between the per-pair clip and
/// the cross-triangle merge.
///
/// Provenance: textbook four-point contact reduction (Ericson, *Real-Time
/// Collision Detection*, 2004). No Unreal Engine source or derived code.
#[must_use]
pub(crate) fn reduce_to_four(points: &[ManifoldPoint], normal: Vec3) -> Vec<ManifoldPoint> {
    if points.len() <= MAX_MANIFOLD_POINTS {
        return points.to_vec();
    }
    // Deepest corner anchors the quad.
    let mut i0 = 0;
    for (i, pt) in points.iter().enumerate() {
        if pt.depth > points[i0].depth {
            i0 = i;
        }
    }
    // Corner farthest from the anchor.
    let p0 = points[i0].position;
    let mut i1 = i0;
    let mut best_d2 = -1.0;
    for (i, pt) in points.iter().enumerate() {
        let d2 = (pt.position - p0).length_squared();
        if d2 > best_d2 {
            best_d2 = d2;
            i1 = i;
        }
    }
    let p1 = points[i1].position;
    let diag = p1 - p0;
    // Corners that maximise signed area on either side of the diagonal.
    let mut i2 = i0;
    let mut i3 = i0;
    let mut best_pos = 0.0f32;
    let mut best_neg = 0.0f32;
    for (i, pt) in points.iter().enumerate() {
        let area = diag.cross(pt.position - p0).dot(normal);
        if area > best_pos {
            best_pos = area;
            i2 = i;
        } else if area < best_neg {
            best_neg = area;
            i3 = i;
        }
    }
    let mut chosen = Vec::with_capacity(MAX_MANIFOLD_POINTS);
    for &idx in &[i0, i1, i2, i3] {
        if !chosen.contains(&idx) {
            chosen.push(idx);
        }
    }
    chosen.iter().map(|&i| points[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-4;

    #[test]
    fn reduce_passes_through_four_or_fewer() {
        let pts = vec![
            ManifoldPoint::new(Vec3::new(0.0, 0.0, 0.0), 0.1),
            ManifoldPoint::new(Vec3::new(1.0, 0.0, 0.0), 0.2),
        ];
        let out = reduce_to_four(&pts, Vec3::Z);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], pts[0]);
        assert_eq!(out[1], pts[1]);
    }

    #[test]
    fn reduce_keeps_bounding_quad_of_a_hexagon() {
        let pts = vec![
            ManifoldPoint::new(Vec3::new(-2.0, 0.0, 0.0), 0.3),
            ManifoldPoint::new(Vec3::new(-1.0, 1.0, 0.0), 0.1),
            ManifoldPoint::new(Vec3::new(1.0, 1.0, 0.0), 0.1),
            ManifoldPoint::new(Vec3::new(2.0, 0.0, 0.0), 0.5),
            ManifoldPoint::new(Vec3::new(1.0, -1.0, 0.0), 0.1),
            ManifoldPoint::new(Vec3::new(-1.0, -1.0, 0.0), 0.1),
        ];
        let out = reduce_to_four(&pts, Vec3::Z);
        assert_eq!(out.len(), 4);
        // The deepest corner (2, 0) and the farthest from it (-2, 0) survive.
        assert!(out.iter().any(|p| (p.position - Vec3::new(2.0, 0.0, 0.0)).length() < EPS));
        assert!(out.iter().any(|p| (p.position - Vec3::new(-2.0, 0.0, 0.0)).length() < EPS));
    }
}
