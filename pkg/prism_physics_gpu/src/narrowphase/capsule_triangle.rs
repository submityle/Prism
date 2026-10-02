//! Capsule-versus-triangle contact geometry, shared bit-for-bit with the `WGSL`
//! kernel.
//!
//! [`capsule_triangle_contact`] is the single source of truth for the
//! capsule-against-triangle collision test and manifold construction; the `CPU`
//! twin ([`cpu_capsule_triangle_narrowphase`]) and the device kernel
//! (`shaders/narrowphase_capsule_triangle.wgsl`) both run this exact arithmetic
//! so their contacts agree to within the floating-point tolerance the parity
//! test allows (only the square root and the handful of reciprocals in the
//! segment/barycentric clamps differ in their low bits).
//!
//! This is the capsule slice of triangle-mesh collision: a static triangle is
//! the atom a trimesh or heightfield collider is built from, so a swept-sphere
//! (capsule) character controller resting on arbitrary level geometry reduces to
//! a batch of capsule-versus-triangle tests, one per candidate triangle the
//! broad phase surfaces. It builds directly on the sphere slice's shared
//! [`closest_point_on_triangle`](super::sphere_triangle::closest_point_on_triangle)
//! Voronoi-region kernel so the point-on-triangle logic lives in exactly one
//! place.
//!
//! # Geometry
//!
//! A [`Capsule`] is a line segment `(p0, p1)` swept by a radius `rc`; a
//! [`Triangle`] is three world-space vertices `a`, `b`, `c`. The test collapses
//! to a sphere-against-triangle test placed at the point `s` on the capsule
//! segment closest to the triangle, versus the point `q` on the solid triangle
//! closest to the segment. Finding `(s, q)` is a segment-versus-triangle
//! closest-feature query:
//!
//! 1. **Pierce test.** If the segment crosses the triangle's plane inside the
//!    triangle, the axis passes through the face: `s == q == hit`, the squared
//!    distance is zero, and the pair is treated as a (degenerate) deep contact.
//!    The plane crossing solves `t = dot(a - p0, n) / dot(p1 - p0, n)` for the
//!    unnormalised face normal `n = (b - a) x (c - a)`, clamped to the segment
//!    by rejecting `t` outside `[0, 1]`; the hit point is then classified
//!    inside the triangle with barycentric coordinates.
//! 2. **Closest feature.** Otherwise the minimum distance is the smallest of
//!    five candidates, evaluated in a fixed order with a strict `<` update: the
//!    two segment endpoints projected onto the solid triangle with
//!    [`closest_point_on_triangle`](super::sphere_triangle::closest_point_on_triangle),
//!    and the capsule segment against each of the three triangle edges with the
//!    clamped segment-segment routine. The winner yields `s` on the segment and
//!    `q` on the triangle.
//!
//! With `(s, q, d2)` in hand the manifold is a sphere-against-triangle contact:
//!
//! * **Axis off the triangle** (`d2 > `[`COINCIDENT_EPS2`] and not pierced):
//!   `dist = sqrt(d2)`; the pair contacts only when `dist < rc` (strict, so a
//!   capsule grazing the surface at `dist == rc` is *not* a contact). The normal
//!   is `(s - q) / dist`, the penetration is `depth = rc - dist`, and the
//!   contact point is the nearest triangle point `q`.
//! * **Axis on or through the triangle** (`pierced` or
//!   `d2 <= `[`COINCIDENT_EPS2`]): the offset `s - q` is degenerate, so the
//!   normal falls back to the triangle's geometric face normal
//!   `normalize((b - a) x (c - a))`, oriented toward the capsule-segment
//!   midpoint so it pushes the capsule off whichever side it sits on. The
//!   contact point is `q`.
//!
//! # Depth of a pierced axis — known limitation
//!
//! When the capsule axis pierces the triangle the true overlap of the swept
//! volume can exceed `rc`, but this slice reports a single-point manifold with
//! `depth = rc` along the face normal. This is a deliberate, documented
//! simplification (a conservative one-point push-out), not a stub: a solver
//! applying it still separates the bodies, and the deep-penetration manifold a
//! trimesh collider ultimately needs is produced by the aggregating manifold
//! stage, not this per-triangle primitive. The arithmetic that *is* reported is
//! exact and identical on both paths.
//!
//! # Normal convention
//!
//! The contact normal points **from the triangle toward the capsule**, i.e. the
//! direction that pushes the capsule off the triangle. The reported [`Contact`]
//! stores the capsule index in `a` and the triangle index in `b`; like the
//! sphere-versus-triangle pair, this type's normal runs `b` (triangle) to `a`
//! (capsule), matching the solver's push-out convention for a dynamic capsule
//! against a static triangle.
//!
//! # Per-operation agreement
//!
//! Every step above runs in the identical order on both paths: the same plane
//! crossing and barycentric inside test, the same five-candidate cascade with
//! its fixed evaluation order and strict `<` updates, the same shared
//! Voronoi-region closest-point routine, the same clamped segment-segment
//! routine, the same strict `dist < rc` overlap test, and the same degenerate
//! fallback to the oriented face normal. Only `sqrt` and the handful of
//! reciprocals are inexact, so the parity test matches the validity flag exactly
//! and the normal, depth, and point within a tight tolerance.
//!
//! Provenance: closest-point-on-triangle is the Voronoi-region method from
//! Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
//! clamped segment-segment routine is Ericson section 5.1.9 (replicated here
//! operation for operation from the capsule-capsule slice, whose copy is
//! private); the segment-triangle pierce test is the textbook plane-crossing
//! plus barycentric classification. No Unreal Engine source or derived code.

use glam::Vec3;

use super::capsule::Capsule;
use super::contact::Contact;
use super::sphere_triangle::{closest_point_on_triangle, Triangle};

/// Squared-length threshold below which the segment-segment and plane-crossing
/// denominators are treated as zero (parallel or degenerate), taking a stable
/// fallback rather than dividing by a near-zero. Kept identical to the `WGSL`
/// constant so both paths branch on the same pairs.
pub(crate) const SEG_EPS2: f32 = 1.0e-12;

/// Squared-distance threshold below which the offset from the nearest triangle
/// point to the nearest capsule-axis point is treated as zero, i.e. the axis
/// lies on the triangle and the face-normal fallback runs. Kept identical to the
/// `WGSL` constant so both paths pick the fallback on the same pairs.
pub(crate) const COINCIDENT_EPS2: f32 = 1.0e-12;

/// A candidate capsule-versus-triangle pair: an index into the capsule slice and
/// an index into the triangle slice.
///
/// The two indices address different arrays, so the pair is intrinsically
/// ordered (`capsule` is always the `a` side, `triangle` the `b` side) and is
/// not canonicalised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapsuleTrianglePair {
    /// Index into the capsule slice; becomes `Contact::a`.
    pub capsule: u32,
    /// Index into the triangle slice; becomes `Contact::b`.
    pub triangle: u32,
}

impl CapsuleTrianglePair {
    /// Creates a capsule-versus-triangle candidate pair.
    #[must_use]
    pub fn new(capsule: u32, triangle: u32) -> CapsuleTrianglePair {
        CapsuleTrianglePair { capsule, triangle }
    }
}

/// Closest point pair between segment `(p1, q1)` and segment `(p2, q2)`.
///
/// Returns `(ca, cb)`, the points on the first and second segment respectively
/// that minimise the distance between the segments. This is the clamped Ericson
/// `ClosestPtSegmentSegment` routine: every division is guarded by
/// [`SEG_EPS2`], so parallel segments (zero denominator) and degenerate
/// zero-length segments both take a stable fallback rather than dividing by
/// zero.
///
/// Replicated operation for operation from the capsule-capsule slice's private
/// copy (`narrowphase/capsule_capsule.rs`) so the capsule-triangle `CPU` and
/// `GPU` paths share identical arithmetic; see that module and Ericson section
/// 5.1.9 for the derivation.
#[must_use]
fn closest_pt_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let s: f32;
    let t: f32;
    if a <= SEG_EPS2 && e <= SEG_EPS2 {
        // Both segments degenerate to points.
        s = 0.0;
        t = 0.0;
    } else if a <= SEG_EPS2 {
        // First segment degenerate: project its point onto the second.
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= SEG_EPS2 {
            // Second segment degenerate: project its point onto the first.
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // General case: solve the 2x2 system, then clamp t back into range
            // and recompute s for a clamped t.
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            // Parallel segments give a zero denominator; pick s = 0 and let the
            // t recomputation below place the closest point.
            let s0 = if denom > SEG_EPS2 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                t = t0;
                s = s0;
            }
        }
    }

    let ca = p1 + d1 * s;
    let cb = p2 + d2 * t;
    (ca, cb)
}

/// The outcome of the segment-versus-triangle closest-feature query: the point
/// `s` on the capsule segment, the point `q` on the solid triangle, their
/// squared distance `d2`, and whether the segment pierced the triangle face.
#[derive(Clone, Copy, Debug)]
struct SegmentTriangle {
    /// Point on the capsule segment nearest the triangle.
    s: Vec3,
    /// Point on the solid triangle nearest the segment.
    q: Vec3,
    /// Squared distance between `s` and `q`.
    d2: f32,
    /// Whether the segment crosses the triangle face (so `s == q == hit`).
    pierced: bool,
}

/// Closest feature between the capsule segment `(p0, p1)` and the solid triangle
/// `(a, b, c)`.
///
/// First tests whether the segment pierces the triangle face by crossing its
/// supporting plane inside the triangle; if so, returns that hit point with a
/// zero squared distance and `pierced = true`. Otherwise it returns the minimum
/// over five candidates — each segment endpoint projected onto the solid
/// triangle, and the segment against each of the three triangle edges —
/// evaluated in a fixed order with a strict `<` update so both paths pick the
/// same winner on a tie.
#[must_use]
fn closest_segment_triangle(p0: Vec3, p1: Vec3, a: Vec3, b: Vec3, c: Vec3) -> SegmentTriangle {
    // --- Pierce test: does the segment cross the triangle plane inside it? ---
    let n = (b - a).cross(c - a);
    let pq = p1 - p0;
    let denom = pq.dot(n);
    if denom.abs() > SEG_EPS2 {
        let t = (a - p0).dot(n) / denom;
        if (0.0..=1.0).contains(&t) {
            let hit = p0 + pq * t;
            // Barycentric inside test for `hit` within triangle (a, b, c).
            let v0 = b - a;
            let v1 = c - a;
            let v2 = hit - a;
            let d00 = v0.dot(v0);
            let d01 = v0.dot(v1);
            let d11 = v1.dot(v1);
            let d20 = v2.dot(v0);
            let d21 = v2.dot(v1);
            let bary_denom = d00 * d11 - d01 * d01;
            if bary_denom.abs() > SEG_EPS2 {
                let v = (d11 * d20 - d01 * d21) / bary_denom;
                let w = (d00 * d21 - d01 * d20) / bary_denom;
                let u = 1.0 - v - w;
                if u >= 0.0 && v >= 0.0 && w >= 0.0 {
                    return SegmentTriangle {
                        s: hit,
                        q: hit,
                        d2: 0.0,
                        pierced: true,
                    };
                }
            }
        }
    }

    // --- Minimum over the five closest-feature candidates. ---
    // Candidate 1: segment endpoint p0 against the solid triangle.
    let q0 = closest_point_on_triangle(p0, a, b, c);
    let mut best_s = p0;
    let mut best_q = q0;
    let mut best_d2 = (p0 - q0).length_squared();

    // Candidate 2: segment endpoint p1 against the solid triangle.
    let q1 = closest_point_on_triangle(p1, a, b, c);
    let d2_1 = (p1 - q1).length_squared();
    if d2_1 < best_d2 {
        best_s = p1;
        best_q = q1;
        best_d2 = d2_1;
    }

    // Candidates 3-5: the segment against each triangle edge, in the fixed
    // order (a, b), (b, c), (c, a).
    let edges = [(a, b), (b, c), (c, a)];
    for (e0, e1) in edges {
        let (cs, cq) = closest_pt_segment_segment(p0, p1, e0, e1);
        let d2_e = (cs - cq).length_squared();
        if d2_e < best_d2 {
            best_s = cs;
            best_q = cq;
            best_d2 = d2_e;
        }
    }

    SegmentTriangle {
        s: best_s,
        q: best_q,
        d2: best_d2,
        pierced: false,
    }
}

/// Tests whether the capsule penetrates the triangle and, if so, builds the
/// contact.
///
/// Returns [`Some`] with the manifold for the pair `(capsule_id, tri_id)` when
/// the capsule overlaps the triangle, or [`None`] when it is separated or
/// exactly touching (a grazing `dist == rc`). The arithmetic mirrors
/// `narrowphase_capsule_triangle.wgsl` operation for operation; see the module
/// documentation for the geometry and the normal convention.
#[must_use]
pub(crate) fn capsule_triangle_contact(
    capsule_id: u32,
    tri_id: u32,
    cap: &Capsule,
    tri: &Triangle,
) -> Option<Contact> {
    let feature = closest_segment_triangle(cap.p0, cap.p1, tri.a, tri.b, tri.c);
    let rc = cap.radius;

    if !feature.pierced && feature.d2 > COINCIDENT_EPS2 {
        let dist = feature.d2.sqrt();
        // Strict overlap: a capsule exactly touching the surface carries no
        // penetration, so it is not a contact.
        if dist >= rc {
            return None;
        }
        let normal = (feature.s - feature.q) / dist;
        let depth = rc - dist;
        Some(Contact::new(capsule_id, tri_id, normal, depth, feature.q))
    } else {
        // Axis on or through the triangle: fall back to the geometric face
        // normal, oriented toward the capsule-segment midpoint so it pushes the
        // capsule off whichever side it sits on.
        let raw = tri.raw_normal();
        let len2 = raw.dot(raw);
        let normal = if len2 > COINCIDENT_EPS2 {
            let mid = (cap.p0 + cap.p1) * 0.5;
            let unit = raw / len2.sqrt();
            if unit.dot(mid - tri.a) < 0.0 {
                -unit
            } else {
                unit
            }
        } else {
            // Degenerate triangle: no face normal, so pick a stable axis.
            Vec3::X
        };
        Some(Contact::new(capsule_id, tri_id, normal, rc, feature.q))
    }
}

/// `CPU` golden twin of the capsule-versus-triangle narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`capsule_triangle_contact`] geometry so its output matches the device kernel
/// operation for operation. It emits one slot per input pair, preserving order,
/// which is what lets the parity test line the `GPU` contacts up against this
/// reference index by index: [`Some`] carrying the manifold when the capsule
/// penetrates the triangle, or [`None`] when it is separated or exactly
/// touching. A later scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a pair references a capsule or triangle index outside the
/// corresponding slice, which is never valid output from a broad phase over the
/// same sets.
#[must_use]
pub fn cpu_capsule_triangle_narrowphase(
    capsules: &[Capsule],
    triangles: &[Triangle],
    pairs: &[CapsuleTrianglePair],
) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let cap = &capsules[pair.capsule as usize];
            let tri = &triangles[pair.triangle as usize];
            capsule_triangle_contact(pair.capsule, pair.triangle, cap, tri)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical unit triangle in the z = 0 plane with a +z face normal.
    fn unit_triangle() -> Triangle {
        Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn face_contact_axis_parallel_above_interior() {
        // Capsule axis parallel to the face, hovering 0.3 above the interior:
        // nearest point is directly below on the face, so the normal is +z and
        // depth = rc - dist = 0.5 - 0.3 = 0.2.
        let tri = unit_triangle();
        let cap = Capsule::new(
            Vec3::new(0.2, 0.3, 0.3),
            Vec3::new(0.4, 0.3, 0.3),
            0.5,
        );
        let c = capsule_triangle_contact(2, 7, &cap, &tri).expect("capsule above face must contact");
        assert_eq!(c.a, 2);
        assert_eq!(c.b, 7);
        assert!((c.normal - Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.2).abs() < 1.0e-6);
        assert!((c.point.z).abs() < 1.0e-6);
    }

    #[test]
    fn edge_contact_against_edge_ab() {
        // Capsule parallel to and below edge AB (the y = 0 edge) at y = -0.3:
        // nearest feature is the edge, normal -y, depth 0.5 - 0.3 = 0.2.
        let tri = unit_triangle();
        let cap = Capsule::new(
            Vec3::new(0.3, -0.3, 0.0),
            Vec3::new(0.7, -0.3, 0.0),
            0.5,
        );
        let c = capsule_triangle_contact(0, 0, &cap, &tri).expect("capsule off edge AB must contact");
        assert!((c.normal - Vec3::new(0.0, -1.0, 0.0)).length() < 1.0e-6);
        assert!((c.depth - 0.2).abs() < 1.0e-6);
        assert!((c.point.y).abs() < 1.0e-6);
    }

    #[test]
    fn vertex_contact_off_vertex_a() {
        // Capsule endpoint sits off vertex A along -x/-y, the other endpoint is
        // farther away, so the nearest feature is the vertex (0, 0, 0).
        let tri = unit_triangle();
        let cap = Capsule::new(
            Vec3::new(-0.3, -0.4, 0.0),
            Vec3::new(-0.9, -1.0, 0.0),
            0.6,
        );
        let c = capsule_triangle_contact(0, 0, &cap, &tri).expect("capsule off vertex A must contact");
        assert!((c.point - Vec3::ZERO).length() < 1.0e-6);
        let diff = Vec3::new(-0.3, -0.4, 0.0);
        let dist = diff.length();
        assert!((c.normal - diff / dist).length() < 1.0e-6);
        assert!((c.depth - (0.6 - dist)).abs() < 1.0e-6);
    }

    #[test]
    fn pierce_through_face_uses_face_normal() {
        // Capsule axis crosses the face from +z to -z through the interior: the
        // segment pierces, so the fallback face normal (oriented toward the
        // segment midpoint on the +z side) is used with depth = rc.
        let tri = unit_triangle();
        let cap = Capsule::new(
            Vec3::new(0.25, 0.25, 0.5),
            Vec3::new(0.25, 0.25, -0.3),
            0.4,
        );
        let c = capsule_triangle_contact(0, 0, &cap, &tri).expect("piercing capsule must contact");
        // Midpoint z = 0.1 > 0, so the +z face normal is kept.
        assert!((c.normal - Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.4).abs() < 1.0e-6);
    }

    #[test]
    fn pierce_from_below_flips_face_normal() {
        // Capsule crosses the face but its midpoint sits on the -z side, so the
        // oriented face normal flips to -z (push the capsule further down).
        let tri = unit_triangle();
        let cap = Capsule::new(
            Vec3::new(0.25, 0.25, 0.3),
            Vec3::new(0.25, 0.25, -0.5),
            0.4,
        );
        let c = capsule_triangle_contact(0, 0, &cap, &tri).expect("piercing capsule must contact");
        // Midpoint z = -0.1 < 0, so the normal flips to -z.
        assert!((c.normal + Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.4).abs() < 1.0e-6);
    }

    #[test]
    fn axis_lying_on_face_falls_back_to_face_normal() {
        // Capsule axis lies in the face plane and over the interior: the nearest
        // distance is zero, so the degenerate fallback runs with the +z normal
        // and depth = rc.
        let tri = unit_triangle();
        let cap = Capsule::new(
            Vec3::new(0.2, 0.3, 0.0),
            Vec3::new(0.4, 0.3, 0.0),
            0.5,
        );
        let c = capsule_triangle_contact(4, 9, &cap, &tri).expect("axis on face always contacts");
        assert!((c.normal - Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn clear_separation_reports_no_contact() {
        // Capsule far above the face: dist 2, radius 0.5, no overlap.
        let tri = unit_triangle();
        let cap = Capsule::new(
            Vec3::new(0.2, 0.3, 2.0),
            Vec3::new(0.4, 0.3, 2.0),
            0.5,
        );
        assert!(capsule_triangle_contact(0, 0, &cap, &tri).is_none());
    }

    #[test]
    fn exactly_touching_reports_no_contact() {
        // Capsule exactly grazing the face at dist == rc: strict test rejects.
        let tri = unit_triangle();
        let cap = Capsule::new(
            Vec3::new(0.2, 0.3, 0.5),
            Vec3::new(0.4, 0.3, 0.5),
            0.5,
        );
        assert!(capsule_triangle_contact(0, 0, &cap, &tri).is_none());
    }

    #[test]
    fn degenerate_triangle_falls_back_to_stable_axis() {
        // A collinear (zero-area) triangle has no face normal; an axis lying on
        // it takes the +x fallback rather than producing a NaN normal.
        let degenerate = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        );
        let cap = Capsule::new(
            Vec3::new(0.4, 0.0, 0.0),
            Vec3::new(0.6, 0.0, 0.0),
            0.5,
        );
        let c = capsule_triangle_contact(0, 0, &cap, &degenerate)
            .expect("axis on degenerate triangle still contacts");
        assert_eq!(c.normal, Vec3::X);
        assert!(c.normal.is_finite());
    }

    #[test]
    fn batch_preserves_order_and_indices() {
        // Two capsules and one triangle: a clear face overlap then a clear gap,
        // checked in input order with their pair indices intact.
        let caps = [
            Capsule::new(Vec3::new(0.2, 0.3, 0.3), Vec3::new(0.4, 0.3, 0.3), 0.5),
            Capsule::new(Vec3::new(0.2, 0.3, 3.0), Vec3::new(0.4, 0.3, 3.0), 0.5),
        ];
        let triangles = [unit_triangle()];
        let pairs = [
            CapsuleTrianglePair::new(0, 0),
            CapsuleTrianglePair::new(1, 0),
        ];
        let contacts = cpu_capsule_triangle_narrowphase(&caps, &triangles, &pairs);
        assert_eq!(contacts.len(), 2);
        let first = contacts[0].expect("first capsule overlaps");
        assert_eq!(first.a, 0);
        assert_eq!(first.b, 0);
        assert!((first.normal - Vec3::Z).length() < 1.0e-6);
        assert!(contacts[1].is_none());
    }

    #[test]
    fn empty_pairs_yield_no_contacts() {
        let caps = [Capsule::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
        let triangles = [unit_triangle()];
        assert!(cpu_capsule_triangle_narrowphase(&caps, &triangles, &[]).is_empty());
    }
}
