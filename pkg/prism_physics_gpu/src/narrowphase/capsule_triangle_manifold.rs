//! Capsule-versus-triangle up-to-two-point contact manifold, shared bit-for-bit
//! with the `WGSL` kernel.
//!
//! The sibling [`capsule_triangle`](super::capsule_triangle) slice reports the
//! single deepest-feature contact, which is enough to push a penetrating capsule
//! off a triangle but lets a capsule lying flat on the triangle face *rock*
//! about that one point. This module promotes that contact to a persistent
//! up-to-two-point manifold — the representation a solver needs to hold a
//! resting capsule still, exactly as [`capsule_obb_manifold`](super::capsule_obb_manifold)
//! does for a box face. The `CPU` twin ([`cpu_capsule_triangle_manifold`]) and
//! the device kernel (`shaders/narrowphase_capsule_triangle_manifold.wgsl`) run
//! the identical arithmetic so their manifolds agree to within the
//! floating-point tolerance the parity test allows.
//!
//! # Geometry
//!
//! The manifold is built in three steps, with the triangle's own plane as the
//! reference face:
//!
//! 1. **Deepest contact.** Run the shared single-point test
//!    ([`capsule_triangle_contact`]). A [`None`] there means no penetration, so
//!    the manifold is [`None`] too. The returned [`Contact`] is both the
//!    penetration gate and the honest fallback when a second point cannot be
//!    found.
//! 2. **Reference face.** Take the triangle's geometric face normal
//!    `normalize((b - a) x (c - a))`, oriented toward the capsule-segment
//!    midpoint so it points the way that pushes the capsule off the triangle —
//!    the same oriented normal the single-point fallback uses. A degenerate
//!    (zero-area) triangle has no face, so the manifold collapses to the single
//!    deepest contact.
//! 3. **Segment-triangle clip.** Clip the capsule axis segment, in its own
//!    parameter `t in [0, 1]`, to the triangle's three edge half-planes (each
//!    edge's inward in-plane normal `n x edge`, oriented toward the opposite
//!    vertex) with the Liang-Barsky algorithm. The surviving span `t0 <= t1`
//!    marks the stretch of the capsule axis that lies over the triangle. Each
//!    boundary point is projected onto the triangle plane and its penetration is
//!    `depth = rc - above`, where `above` is the signed height of the axis point
//!    over the plane. Points with a positive depth are live corners.
//!
//! When both boundary points are live and distinct the manifold carries two
//! points sharing the oriented face normal. Otherwise — the clip misses the
//! triangle, collapses to a sliver, or leaves fewer than two penetrating corners
//! — the manifold honestly reports the single deepest contact (`count == 1`),
//! never a fabricated second point.
//!
//! # Normal convention
//!
//! The shared [`normal`](ContactManifold::normal) points **from the triangle
//! toward the capsule** — the direction that pushes the capsule off the
//! triangle — matching the single-point [`capsule_triangle`](super::capsule_triangle)
//! contact. The capsule is the `a` side and the triangle the `b` side of every
//! reported manifold.
//!
//! # Liang-Barsky clip
//!
//! Clipping the segment to the triangle by tracking one entering and one leaving
//! parameter over three half-planes is branch-simple and free of any sort or
//! square root, so the `WGSL` twin reproduces the exact same `t0`/`t1` (only the
//! three reciprocals in the edge-crossing solves differ in their low bits). A
//! segment parallel to and outside an edge rejects the whole clip; parallel and
//! inside adds no constraint.
//!
//! Provenance: closest-point-on-triangle and the four-point reduction are from
//! Christer Ericson, *Real-Time Collision Detection* (2004); the Liang-Barsky
//! segment clip against the triangle's edge half-planes is textbook. No Unreal
//! Engine source or derived code.

use glam::Vec3;

use super::capsule::Capsule;
use super::capsule_triangle::{capsule_triangle_contact, CapsuleTrianglePair};
use super::contact::Contact;
use super::manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};
use super::sphere_triangle::Triangle;

/// Parameter-span threshold below which the clipped stretch of the capsule axis
/// is treated as a single point (the segment grazes a triangle corner), so the
/// manifold collapses to the single deepest contact. Kept identical to the
/// `WGSL` constant so both paths take the collapse on the same segments.
pub(crate) const CLIP_T_EPS: f32 = 1.0e-9;

/// Squared-distance threshold below which the two clipped corners are treated as
/// coincident, collapsing the manifold to a single point. Kept identical to the
/// `WGSL` constant.
pub(crate) const SEP_EPS2: f32 = 1.0e-12;

/// Squared-length threshold below which the triangle's raw face normal is
/// treated as degenerate (zero area), so no reference face exists and the
/// manifold collapses to the single deepest contact. Kept identical to the
/// `WGSL` constant.
pub(crate) const FACE_EPS2: f32 = 1.0e-12;

/// Clips the capsule axis `p0 -> p1` to the triangle `(a, b, c)` edge half-planes
/// in the face plane whose unit normal is `n`, returning the entering and leaving
/// parameters `t0 <= t1` in `[0, 1]`, or [`None`] when the axis lies wholly
/// outside the triangle footprint.
///
/// Each edge contributes a constraint `t * p <= q`, with `p = -slope` and
/// `q = d_enter` for the inward half-plane value `g(t) = d_enter + slope * t >= 0`:
/// a negative `p` is an entering crossing (raises `t0`), a positive `p` a leaving
/// crossing (lowers `t1`), and a zero `p` is a segment parallel to the edge that
/// rejects the clip only when it sits on the outer side (`q < 0`).
#[must_use]
fn clip_segment_to_triangle(
    p0: Vec3,
    p1: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    n: Vec3,
) -> Option<(f32, f32)> {
    let seg = p1 - p0;
    // Each tuple is (edge start, opposite vertex) in the fixed order used on both
    // paths: edge (a, b) opposite c, edge (b, c) opposite a, edge (c, a) opposite b.
    let edges = [(a, b, c), (b, c, a), (c, a, b)];
    let mut t_enter = 0.0f32;
    let mut t_leave = 1.0f32;
    for (e0, e1, opp) in edges {
        // Inward in-plane normal, oriented toward the opposite vertex.
        let mut inward = n.cross(e1 - e0);
        if inward.dot(opp - e0) < 0.0 {
            inward = -inward;
        }
        let d_enter = (p0 - e0).dot(inward);
        let slope = seg.dot(inward);
        let p = -slope;
        let q = d_enter;
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else if p < 0.0 {
            let t = q / p;
            if t > t_leave {
                return None;
            }
            if t > t_enter {
                t_enter = t;
            }
        } else {
            let t = q / p;
            if t < t_enter {
                return None;
            }
            if t < t_leave {
                t_leave = t;
            }
        }
    }
    Some((t_enter, t_leave))
}

/// Wraps a single deepest-feature [`Contact`] as a one-point manifold, the
/// honest fallback whenever a stable second point cannot be found.
#[must_use]
fn single_point(contact: Contact) -> ContactManifold {
    let points = [ManifoldPoint::new(contact.point, contact.depth)];
    ContactManifold::new(contact.a, contact.b, contact.normal, 1, &points)
}

/// Tests whether the capsule penetrates the triangle and, if so, builds the
/// contact manifold at up to two points along the triangle face.
///
/// Returns [`Some`] with a one- or two-point manifold for the pair
/// `(capsule_id, tri_id)` when the capsule overlaps the triangle, or [`None`]
/// when it is separated or exactly touching. A capsule lying flat over the
/// triangle reports two points (one per end of the clipped overlap) so a solver
/// can resist rotation; a capsule touching at a single feature, a sliver clip, or
/// a clip that misses the triangle collapses honestly to the single deepest
/// contact. The arithmetic mirrors `narrowphase_capsule_triangle_manifold.wgsl`
/// operation for operation; see the module documentation for the geometry and
/// the normal convention.
#[must_use]
pub(crate) fn capsule_triangle_manifold(
    capsule_id: u32,
    tri_id: u32,
    cap: &Capsule,
    tri: &Triangle,
) -> Option<ContactManifold> {
    // The single deepest contact is both the penetration gate and the fallback.
    let contact = capsule_triangle_contact(capsule_id, tri_id, cap, tri)?;

    // Reference face: the oriented triangle face normal. A degenerate triangle
    // has no face, so keep the single deepest contact.
    let raw = tri.raw_normal();
    let len2 = raw.dot(raw);
    if len2 <= FACE_EPS2 {
        return Some(single_point(contact));
    }
    let unit = raw / len2.sqrt();
    let mid = (cap.p0 + cap.p1) * 0.5;
    let n = if unit.dot(mid - tri.a) < 0.0 {
        -unit
    } else {
        unit
    };

    // Clip the capsule axis to the triangle footprint in the face plane.
    let Some((t0, t1)) = clip_segment_to_triangle(cap.p0, cap.p1, tri.a, tri.b, tri.c, n) else {
        return Some(single_point(contact));
    };
    if t1 - t0 <= CLIP_T_EPS {
        // The overlap is a sliver: no stable second point, keep the deepest one.
        return Some(single_point(contact));
    }

    // Project each clip-boundary point onto the triangle plane and keep the ones
    // whose swept surface actually dips below the plane.
    let seg = cap.p1 - cap.p0;
    let rc = cap.radius;
    let mut points = [ManifoldPoint::new(Vec3::ZERO, 0.0); MAX_MANIFOLD_POINTS];
    let mut count = 0usize;
    for t in [t0, t1] {
        let axis_pt = cap.p0 + seg * t;
        // Signed height of the axis point above the triangle plane.
        let above = (axis_pt - tri.a).dot(n);
        let depth = rc - above;
        if depth <= 0.0 {
            continue;
        }
        let world = axis_pt - n * above;
        points[count] = ManifoldPoint::new(world, depth);
        count += 1;
    }

    if count < 2 {
        // Only one end (or neither) clears the plane: fall back to the deepest
        // contact rather than invent a second point.
        return Some(single_point(contact));
    }
    // Reject a degenerate manifold whose two corners coincide.
    let sep = points[1].position - points[0].position;
    if sep.dot(sep) <= SEP_EPS2 {
        return Some(single_point(contact));
    }

    Some(ContactManifold::new(capsule_id, tri_id, n, count, &points))
}

/// `CPU` golden twin of the capsule-versus-triangle manifold narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`capsule_triangle_manifold`] geometry so its output matches the device
/// kernel operation for operation. It emits one slot per input pair, preserving
/// order, which is what lets the parity test line the `GPU` manifolds up against
/// this reference index by index: [`Some`] carrying the one- or two-point
/// manifold when the capsule penetrates the triangle, or [`None`] when it is
/// separated or exactly touching. A later scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a pair references a capsule or triangle index outside the
/// corresponding slice, which is never valid output from a broad phase over the
/// same sets.
#[must_use]
pub fn cpu_capsule_triangle_manifold(
    capsules: &[Capsule],
    triangles: &[Triangle],
    pairs: &[CapsuleTrianglePair],
) -> Vec<Option<ContactManifold>> {
    pairs
        .iter()
        .map(|pair| {
            let cap = &capsules[pair.capsule as usize];
            let tri = &triangles[pair.triangle as usize];
            capsule_triangle_manifold(pair.capsule, pair.triangle, cap, tri)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sphere_triangle::Triangle;

    /// A capsule from two endpoints and a radius.
    fn cap(p0: Vec3, p1: Vec3, radius: f32) -> Capsule {
        Capsule::new(p0, p1, radius)
    }

    /// A large triangle on the `z = 0` plane whose geometric normal is `+z`.
    fn floor() -> Triangle {
        Triangle::new(
            Vec3::new(-10.0, -10.0, 0.0),
            Vec3::new(10.0, -10.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
        )
    }

    #[test]
    fn flat_capsule_over_face_reports_two_points() {
        // Capsule lying flat along +x at height z = 0.4, radius 0.5: both ends
        // (x = -2 and x = +2, well inside the triangle footprint) sit 0.4 above
        // the face, so depth = 0.5 - 0.4 = 0.1 at each clipped corner.
        let triangles = [floor()];
        let capsules = [cap(Vec3::new(-2.0, 0.0, 0.4), Vec3::new(2.0, 0.0, 0.4), 0.5)];
        let pairs = [CapsuleTrianglePair::new(0, 0)];
        let m = cpu_capsule_triangle_manifold(&capsules, &triangles, &pairs)[0]
            .expect("flat overlap");
        assert_eq!(m.a, 0);
        assert_eq!(m.b, 0);
        assert_eq!(m.count, 2);
        assert!((m.normal - Vec3::Z).length() < 1.0e-6, "normal {:?}", m.normal);
        assert!((m.points[0].depth - 0.1).abs() < 1.0e-5, "d0 {}", m.points[0].depth);
        assert!((m.points[1].depth - 0.1).abs() < 1.0e-5, "d1 {}", m.points[1].depth);
        // Both corners project onto the face plane z = 0 at the clipped ends.
        assert!(m.points[0].position.z.abs() < 1.0e-5);
        assert!(m.points[1].position.z.abs() < 1.0e-5);
        let xs = [m.points[0].position.x, m.points[1].position.x];
        assert!(xs.iter().any(|&x| (x + 2.0).abs() < 1.0e-5), "no x=-2 corner: {xs:?}");
        assert!(xs.iter().any(|&x| (x - 2.0).abs() < 1.0e-5), "no x=+2 corner: {xs:?}");
    }

    #[test]
    fn vertical_capsule_through_face_collapses_to_one_point() {
        // A capsule standing on the plane normal pierces the face at a single
        // point: only the lower end clears the plane, so the manifold honestly
        // reports one point rather than inventing a second.
        let triangles = [floor()];
        let capsules = [cap(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 0.5)];
        let pairs = [CapsuleTrianglePair::new(0, 0)];
        let m = cpu_capsule_triangle_manifold(&capsules, &triangles, &pairs)[0]
            .expect("piercing overlap");
        assert_eq!(m.count, 1);
    }

    #[test]
    fn tilted_capsule_one_end_lifted_falls_back_to_one_point() {
        // One end hugs the face (penetrating) while the other rides far above
        // it: only one clipped corner is live, so the manifold falls back to the
        // single deepest contact.
        let triangles = [floor()];
        let capsules = [cap(Vec3::new(0.0, 0.0, 0.4), Vec3::new(0.0, 0.0, 3.0), 0.5)];
        let pairs = [CapsuleTrianglePair::new(0, 0)];
        let m = cpu_capsule_triangle_manifold(&capsules, &triangles, &pairs)[0]
            .expect("one-end overlap");
        assert_eq!(m.count, 1);
    }

    #[test]
    fn separated_capsule_reports_none() {
        // A capsule hovering well above the face never penetrates.
        let triangles = [floor()];
        let capsules = [cap(Vec3::new(-2.0, 0.0, 5.0), Vec3::new(2.0, 0.0, 5.0), 0.5)];
        let pairs = [CapsuleTrianglePair::new(0, 0)];
        assert!(cpu_capsule_triangle_manifold(&capsules, &triangles, &pairs)[0].is_none());
    }

    #[test]
    fn degenerate_triangle_never_reports_two_points() {
        // A zero-area (collinear) triangle has no reference face, so a touching
        // capsule can only ever report the single deepest contact, never two.
        let triangles = [Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        )];
        let capsules = [cap(Vec3::new(0.5, 0.1, 0.0), Vec3::new(1.5, 0.1, 0.0), 0.5)];
        let pairs = [CapsuleTrianglePair::new(0, 0)];
        if let Some(m) = cpu_capsule_triangle_manifold(&capsules, &triangles, &pairs)[0] {
            assert_eq!(m.count, 1, "degenerate triangle must collapse to one point");
        }
    }

    #[test]
    fn empty_batch_yields_no_manifolds() {
        let triangles = [floor()];
        let capsules = [cap(Vec3::new(-2.0, 0.0, 0.4), Vec3::new(2.0, 0.0, 0.4), 0.5)];
        let pairs: [CapsuleTrianglePair; 0] = [];
        assert!(cpu_capsule_triangle_manifold(&capsules, &triangles, &pairs).is_empty());
    }
}
