//! Capsule-versus-OBB two-point contact manifold, shared bit-for-bit with the
//! `WGSL` kernel.
//!
//! The sibling [`capsule_obb`](super::capsule_obb) slice reports the single
//! deepest-feature contact, which is enough to push a penetrating capsule out of
//! a box but lets a capsule lying flat on a box face *rock* about that one
//! point. This module promotes that contact to a persistent up-to-two-point
//! manifold — the representation a solver needs to hold a resting capsule still,
//! exactly as [`capsule_halfspace`](super::capsule_halfspace) does for the
//! infinite-plane case. The `CPU` twin ([`cpu_capsule_obb_manifold`]) and the
//! device kernel (`shaders/narrowphase_capsule_obb_manifold.wgsl`) run the
//! identical arithmetic so their manifolds agree to within the floating-point
//! tolerance the parity test allows.
//!
//! # Geometry
//!
//! The manifold is built in three steps, all in the box's local frame where the
//! box is the axis-aligned box `[-he, he]`:
//!
//! 1. **Deepest contact.** Run the shared single-point test
//!    ([`capsule_obb_contact`]). A [`None`] there means no penetration, so the
//!    manifold is [`None`] too. The returned [`Contact`] is both the source of
//!    the reference face and the honest fallback when a second point cannot be
//!    found.
//! 2. **Reference face.** Project the contact normal into the box frame; the
//!    axis `k` it aligns with most (largest `|component|`) picks the box face,
//!    and its sign `s` picks the `+`/`-` side. The face lives in the plane
//!    `local[k] == s * he[k]`, spanned by the other two axes `u = (k + 1) % 3`
//!    and `v = (k + 2) % 3`.
//! 3. **Segment-rectangle clip.** Project both capsule endpoints into the frame
//!    and clip the axis segment, in the `(u, v)` face coordinates, to the face
//!    rectangle `[-he[u], he[u]] x [-he[v], he[v]]` with the Liang-Barsky
//!    algorithm. The two clip-boundary parameters `t0 <= t1` give the stretch of
//!    the capsule axis that lies over the face. Each boundary point is projected
//!    onto the face (its `k` coordinate snapped to `s * he[k]`) and its
//!    penetration is `depth = rc - (s * local[k] - he[k])`: the swept radius less
//!    how far the axis point sits above the face plane. Points with a positive
//!    depth are live corners.
//!
//! When both boundary points are live and distinct, the manifold carries two
//! points sharing the reference-face outward normal `axis[k] * s`. Otherwise —
//! the clip misses the face, collapses to a sliver, or leaves fewer than two
//! penetrating corners — the manifold honestly reports the single deepest
//! contact (`count == 1`), never a fabricated second point.
//!
//! # Normal convention
//!
//! The shared [`normal`](ContactManifold::normal) is the reference face's
//! outward normal — the direction that pushes the capsule out of the box —
//! matching the single-point [`capsule_obb`](super::capsule_obb) contact and the
//! [`capsule_halfspace`](super::capsule_halfspace) manifold. The capsule is the
//! `a` side and the box the `b` side of every reported manifold.
//!
//! # Liang-Barsky clip
//!
//! Clipping a segment to an axis-aligned rectangle by tracking one entering and
//! one leaving parameter is branch-simple and free of any sort or square root,
//! so the `WGSL` twin reproduces the exact same `t0`/`t1` (only the four
//! reciprocals in the edge-crossing solves differ in their low bits). A segment
//! parallel to and outside an edge rejects the whole clip; parallel and inside
//! adds no constraint.
//!
//! Provenance: textbook capsule-versus-oriented-bounding-box manifold
//! (reference-face selection plus Liang-Barsky segment-rectangle clipping); no
//! Unreal Engine source or derived code.

use glam::Vec3;

use super::capsule::Capsule;
use super::capsule_obb::{capsule_obb_contact, CapsuleObbPair};
use super::contact::Contact;
use super::manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};
use super::obb::Obb;

/// Parameter-span threshold below which the clipped stretch of the capsule axis
/// is treated as a single point (the segment grazes a face corner), so the
/// manifold collapses to the single deepest contact. Kept identical to the
/// `WGSL` constant so both paths take the collapse on the same segments.
pub(crate) const CLIP_T_EPS: f32 = 1.0e-9;

/// Squared-distance threshold below which the two clipped corners are treated as
/// coincident, collapsing the manifold to a single point. Kept identical to the
/// `WGSL` constant.
pub(crate) const SEP_EPS2: f32 = 1.0e-12;

/// The reference-face axis `k`, plus the two in-plane axes `u` and `v`.
///
/// `k` is the axis the local contact normal aligns with most strongly (largest
/// `|component|`, ties resolving to the lower axis index), and `u`, `v` are the
/// remaining axes in cyclic order `(k + 1) % 3`, `(k + 2) % 3`. Fixing the
/// tie-break keeps the `WGSL` twin on the same face for a normal that splits two
/// axes evenly.
#[must_use]
fn reference_axes(n_local: Vec3) -> (usize, usize, usize) {
    let ax = n_local.x.abs();
    let ay = n_local.y.abs();
    let az = n_local.z.abs();
    let mut k = 0usize;
    let mut best = ax;
    if ay > best {
        best = ay;
        k = 1;
    }
    if az > best {
        k = 2;
    }
    (k, (k + 1) % 3, (k + 2) % 3)
}

/// Component `i` of `v` (`0 -> x`, `1 -> y`, else `z`).
///
/// A helper so the axis-indexed arithmetic reads the same on both paths; `glam`
/// has no runtime component index.
#[must_use]
fn comp(v: Vec3, i: usize) -> f32 {
    match i {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Clips the segment `p0 -> p1`, given in one face's `(u, v)` coordinates, to the
/// rectangle `[-he_u, he_u] x [-he_v, he_v]` with the Liang-Barsky algorithm.
///
/// Returns the entering and leaving parameters `t0 <= t1` in `[0, 1]` when the
/// segment meets the rectangle, or [`None`] when it lies wholly outside. Each of
/// the four edges contributes a constraint `t * p <= q`: a negative `p` is an
/// entering crossing (raises `t0`), a positive `p` a leaving crossing (lowers
/// `t1`), and a zero `p` is a parallel edge that rejects the clip only when the
/// segment sits on its outer side (`q < 0`).
#[must_use]
fn clip_segment_to_rect(
    p0u: f32,
    p0v: f32,
    p1u: f32,
    p1v: f32,
    he_u: f32,
    he_v: f32,
) -> Option<(f32, f32)> {
    let du = p1u - p0u;
    let dv = p1v - p0v;
    let edges = [
        (-du, p0u + he_u), // left:   u >= -he_u
        (du, he_u - p0u),  // right:  u <=  he_u
        (-dv, p0v + he_v), // bottom: v >= -he_v
        (dv, he_v - p0v),  // top:    v <=  he_v
    ];
    let mut t_enter = 0.0f32;
    let mut t_leave = 1.0f32;
    for (p, q) in edges {
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

/// Tests whether the capsule penetrates the box and, if so, builds the contact
/// manifold at up to two points along the reference face.
///
/// Returns [`Some`] with a one- or two-point manifold for the couple
/// `(capsule_id, obb_id)` when the capsule overlaps the box, or [`None`] when it
/// is separated or exactly touching. A capsule lying flat over a box face
/// reports two points (one per end of the clipped face overlap) so a solver can
/// resist rotation; a capsule touching at a single feature, a sliver clip, or a
/// clip that misses the face collapses honestly to the single deepest contact.
/// The arithmetic mirrors `narrowphase_capsule_obb_manifold.wgsl` operation for
/// operation; see the module documentation for the geometry and the normal
/// convention.
#[must_use]
pub(crate) fn capsule_obb_manifold(
    capsule_id: u32,
    obb_id: u32,
    cap: &Capsule,
    box_: &Obb,
) -> Option<ContactManifold> {
    // The single deepest contact is both the penetration gate and the fallback.
    let contact = capsule_obb_contact(capsule_id, obb_id, cap, box_)?;

    let a0 = box_.axes[0];
    let a1 = box_.axes[1];
    let a2 = box_.axes[2];
    let he = box_.half_extents;
    let rc = cap.radius;

    // Project both capsule endpoints into the box frame.
    let d0 = cap.p0 - box_.center;
    let d1 = cap.p1 - box_.center;
    let local0 = Vec3::new(d0.dot(a0), d0.dot(a1), d0.dot(a2));
    let local1 = Vec3::new(d1.dot(a0), d1.dot(a1), d1.dot(a2));

    // Reference face: the axis the contact normal aligns with most, and its side.
    let n_local = Vec3::new(
        contact.normal.dot(a0),
        contact.normal.dot(a1),
        contact.normal.dot(a2),
    );
    let (k, u, v) = reference_axes(n_local);
    let s = if comp(n_local, k) >= 0.0 { 1.0 } else { -1.0 };

    // Clip the capsule axis to the reference face rectangle in the (u, v) plane.
    let Some((t0, t1)) = clip_segment_to_rect(
        comp(local0, u),
        comp(local0, v),
        comp(local1, u),
        comp(local1, v),
        comp(he, u),
        comp(he, v),
    ) else {
        return Some(single_point(contact));
    };
    if t1 - t0 <= CLIP_T_EPS {
        // The overlap is a sliver: no stable second point, keep the deepest one.
        return Some(single_point(contact));
    }

    // Project each clip-boundary point onto the reference face and keep the ones
    // whose swept surface actually dips below the face plane.
    let seg = local1 - local0;
    let he_k = comp(he, k);
    let mut points = [ManifoldPoint::new(Vec3::ZERO, 0.0); MAX_MANIFOLD_POINTS];
    let mut count = 0usize;
    for t in [t0, t1] {
        let local_pt = local0 + seg * t;
        // Signed height of the axis point above the reference face plane.
        let above = s * comp(local_pt, k) - he_k;
        let depth = rc - above;
        if depth <= 0.0 {
            continue;
        }
        let face_k = s * he_k;
        let q = match k {
            0 => Vec3::new(face_k, local_pt.y, local_pt.z),
            1 => Vec3::new(local_pt.x, face_k, local_pt.z),
            _ => Vec3::new(local_pt.x, local_pt.y, face_k),
        };
        let world = box_.center + a0 * q.x + a1 * q.y + a2 * q.z;
        points[count] = ManifoldPoint::new(world, depth);
        count += 1;
    }

    if count < 2 {
        // Only one end (or neither) clears the face: fall back to the deepest
        // contact rather than invent a second point.
        return Some(single_point(contact));
    }
    // Reject a degenerate manifold whose two corners coincide.
    let sep = points[1].position - points[0].position;
    if sep.dot(sep) <= SEP_EPS2 {
        return Some(single_point(contact));
    }

    let face_axis = match k {
        0 => a0,
        1 => a1,
        _ => a2,
    };
    let normal = face_axis * s;
    Some(ContactManifold::new(
        capsule_id, obb_id, normal, count, &points,
    ))
}

/// `CPU` golden twin of the capsule-versus-OBB manifold narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`capsule_obb_manifold`] geometry so its output matches the device kernel
/// operation for operation. It emits one slot per input couple, preserving
/// order, which is what lets the parity test line the `GPU` manifolds up against
/// this reference index by index: [`Some`] carrying the one- or two-point
/// manifold when the capsule penetrates the box, or [`None`] when it is
/// separated or exactly touching. A later scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a pair references a capsule or box index outside the corresponding
/// slice, which is never valid output from a broad phase over the same sets.
#[must_use]
pub fn cpu_capsule_obb_manifold(
    capsules: &[Capsule],
    boxes: &[Obb],
    pairs: &[CapsuleObbPair],
) -> Vec<Option<ContactManifold>> {
    pairs
        .iter()
        .map(|pair| {
            let cap = &capsules[pair.capsule as usize];
            let box_ = &boxes[pair.obb as usize];
            capsule_obb_manifold(pair.capsule, pair.obb, cap, box_)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// A capsule from two endpoints and a radius.
    fn cap(p0: Vec3, p1: Vec3, radius: f32) -> Capsule {
        Capsule::new(p0, p1, radius)
    }

    /// An axis-aligned unit-half-extent box centred at the origin.
    fn unit_box() -> Obb {
        Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::splat(1.0))
    }

    #[test]
    fn flat_capsule_on_top_face_reports_two_points() {
        // A capsule lying flat along +x, radius 0.5, axis at y = 1.4: both ends
        // (clipped to x in [-1, 1]) sit 0.4 above the top face, depth 0.1.
        let boxes = [unit_box()];
        let capsules = [cap(
            Vec3::new(-2.0, 1.4, 0.0),
            Vec3::new(2.0, 1.4, 0.0),
            0.5,
        )];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let m = cpu_capsule_obb_manifold(&capsules, &boxes, &pairs)[0].expect("flat overlap");
        assert_eq!(m.a, 0);
        assert_eq!(m.b, 0);
        assert_eq!(m.count, 2);
        assert!(
            (m.normal - Vec3::Y).length() < 1.0e-6,
            "normal {:?}",
            m.normal
        );
        // Both corners sit on the top face, y = 1.0, at the clipped ends x = -1
        // and x = +1, each with depth 0.1.
        assert!(
            (m.points[0].depth - 0.1).abs() < 1.0e-5,
            "d0 {}",
            m.points[0].depth
        );
        assert!(
            (m.points[1].depth - 0.1).abs() < 1.0e-5,
            "d1 {}",
            m.points[1].depth
        );
        assert!((m.points[0].position.y - 1.0).abs() < 1.0e-5);
        assert!((m.points[1].position.y - 1.0).abs() < 1.0e-5);
        let xs = [m.points[0].position.x, m.points[1].position.x];
        assert!(
            xs.iter().any(|&x| (x + 1.0).abs() < 1.0e-5),
            "no x=-1 corner: {xs:?}"
        );
        assert!(
            xs.iter().any(|&x| (x - 1.0).abs() < 1.0e-5),
            "no x=+1 corner: {xs:?}"
        );
    }

    #[test]
    fn vertical_capsule_poking_side_face_reports_two_points() {
        // A vertical capsule at x = 1.3 spanning y in [0, 3]: the stretch over
        // the +x face is y in [0, 1] (clipped), so two corners on that face.
        let boxes = [unit_box()];
        let capsules = [cap(Vec3::new(1.3, 0.0, 0.0), Vec3::new(1.3, 3.0, 0.0), 0.5)];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let m = cpu_capsule_obb_manifold(&capsules, &boxes, &pairs)[0].expect("side overlap");
        assert_eq!(m.count, 2);
        assert!(
            (m.normal - Vec3::X).length() < 1.0e-6,
            "normal {:?}",
            m.normal
        );
        // x = 1.3, face at 1.0, gap 0.3 < rc 0.5 -> depth 0.2 at both corners.
        assert!((m.points[0].depth - 0.2).abs() < 1.0e-5);
        assert!((m.points[1].depth - 0.2).abs() < 1.0e-5);
        // Both corners sit on the +x face.
        assert!((m.points[0].position.x - 1.0).abs() < 1.0e-5);
        assert!((m.points[1].position.x - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn tilted_capsule_with_one_end_lifted_falls_back_to_one_point() {
        // p0 hugs the top face (depth 0.1); p1 rides far above it (no
        // penetration), so only one clipped corner is live -> single point.
        let boxes = [unit_box()];
        let capsules = [cap(
            Vec3::new(0.0, 1.4, -0.5),
            Vec3::new(0.0, 3.0, 0.5),
            0.5,
        )];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let m = cpu_capsule_obb_manifold(&capsules, &boxes, &pairs)[0].expect("one end buried");
        assert_eq!(m.count, 1);
        // The deepest contact is near p0 on the top face, depth 0.1.
        assert!(
            (m.points[0].depth - 0.1).abs() < 1.0e-5,
            "depth {}",
            m.points[0].depth
        );
    }

    #[test]
    fn capsule_threading_through_falls_back_to_one_point() {
        // A capsule axis passing straight through the box along x at y = 0.2.
        // The closest-feature search keeps the earliest t reaching zero
        // distance, which lands on the entry -x face at local (-1, 0.2, 0):
        // there pen.x = he.x - |x| = 0, the smallest penetration, so the exit
        // face is the -x face (normal -X, depth rc + 0 = 0.25). The reference
        // face is then the -x face, but the capsule axis runs along x, so both
        // endpoints project to the same point on the (y, z) face rectangle and
        // only one clipped corner carries positive depth. The manifold honestly
        // reports a single point rather than fabricating a second one.
        let boxes = [unit_box()];
        let capsules = [cap(
            Vec3::new(-3.0, 0.2, 0.0),
            Vec3::new(3.0, 0.2, 0.0),
            0.25,
        )];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let m = cpu_capsule_obb_manifold(&capsules, &boxes, &pairs)[0].expect("through overlap");
        assert_eq!(m.count, 1);
        assert!(
            (m.normal - (-Vec3::X)).length() < 1.0e-5,
            "normal {:?}",
            m.normal
        );
        // Exit through the -x entry face: depth = rc + (he.x - |x|) = 0.25 + 0.
        assert!(
            (m.points[0].depth - 0.25).abs() < 1.0e-5,
            "d0 {}",
            m.points[0].depth
        );
    }

    #[test]
    fn clear_capsule_reports_none() {
        let boxes = [unit_box()];
        let capsules = [cap(
            Vec3::new(-2.0, 5.0, 0.0),
            Vec3::new(2.0, 5.0, 0.0),
            0.5,
        )];
        let pairs = [CapsuleObbPair::new(0, 0)];
        assert!(cpu_capsule_obb_manifold(&capsules, &boxes, &pairs)[0].is_none());
    }

    #[test]
    fn grazing_touch_reports_none() {
        // Axis at y = 1.5, face at 1.0, gap exactly rc 0.5: the strict
        // single-point gate rejects it, so no manifold either.
        let boxes = [unit_box()];
        let capsules = [cap(
            Vec3::new(-2.0, 1.5, 0.0),
            Vec3::new(2.0, 1.5, 0.0),
            0.5,
        )];
        let pairs = [CapsuleObbPair::new(0, 0)];
        assert!(cpu_capsule_obb_manifold(&capsules, &boxes, &pairs)[0].is_none());
    }

    #[test]
    fn degenerate_capsule_matches_a_single_point() {
        // A zero-length capsule is a sphere: the two clip parameters coincide,
        // so the manifold collapses to one point, matching a sphere-box contact.
        let boxes = [unit_box()];
        let capsules = [cap(Vec3::new(0.0, 1.4, 0.0), Vec3::new(0.0, 1.4, 0.0), 0.5)];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let m = cpu_capsule_obb_manifold(&capsules, &boxes, &pairs)[0].expect("sphere overlap");
        assert_eq!(m.count, 1);
        assert!(
            (m.normal - Vec3::Y).length() < 1.0e-5,
            "normal {:?}",
            m.normal
        );
        assert!(
            (m.points[0].depth - 0.1).abs() < 1.0e-5,
            "depth {}",
            m.points[0].depth
        );
    }

    #[test]
    fn tilted_box_projects_into_the_local_frame() {
        // A box rotated 45 degrees about z with a capsule lying flat along the
        // rotated x axis above the rotated top face: two points, rotated normal.
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let boxes = [Obb::from_quat(Vec3::ZERO, rot, Vec3::splat(1.0))];
        let up = rot * Vec3::Y;
        let along = rot * Vec3::X;
        let centre = up * 1.4;
        let capsules = [cap(centre - along * 2.0, centre + along * 2.0, 0.5)];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let m = cpu_capsule_obb_manifold(&capsules, &boxes, &pairs)[0].expect("tilted overlap");
        assert_eq!(m.count, 2);
        assert!((m.normal - up).length() < 1.0e-5, "normal {:?}", m.normal);
        assert!((m.points[0].depth - 0.1).abs() < 1.0e-5);
        assert!((m.points[1].depth - 0.1).abs() < 1.0e-5);
    }

    #[test]
    fn batch_preserves_input_order() {
        let boxes = [unit_box()];
        let capsules = [
            cap(Vec3::new(-2.0, 1.4, 0.0), Vec3::new(2.0, 1.4, 0.0), 0.5), // two points
            cap(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 0.5), // miss
        ];
        let pairs = [CapsuleObbPair::new(0, 0), CapsuleObbPair::new(1, 0)];
        let out = cpu_capsule_obb_manifold(&capsules, &boxes, &pairs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].expect("first hits").count, 2);
        assert!(out[1].is_none());
    }

    #[test]
    fn empty_pairs_yield_no_manifolds() {
        let capsules = [cap(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
        let boxes = [unit_box()];
        assert!(cpu_capsule_obb_manifold(&capsules, &boxes, &[]).is_empty());
    }
}
