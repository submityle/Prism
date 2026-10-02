//! Multi-point convex-versus-convex contact manifolds via reference-face
//! clipping, built on the shared `GJK`/`EPA` core.
//!
//! [`gjk`](super::gjk::gjk) decides overlap and
//! [`epa`](super::epa::epa) recovers the minimum-translation normal and
//! penetration depth, which together un-penetrate two arbitrary convex
//! polyhedra with a single push-out vector. That is enough to separate the
//! pair but not to hold a face flush against a face without rocking. This
//! module promotes that single penetration into a full manifold: up to
//! [`MAX_MANIFOLD_POINTS`] coplanar corners sharing one normal, the contact
//! a stacking-stable solver consumes.
//!
//! # From penetration vector to a manifold
//!
//! `EPA` yields the push-out normal (pointing from hull `B` toward hull `A`)
//! and the overlap depth along it. From there the manifold is the classic
//! reference/incident face clip used by `Box2D`, Bullet, and Havok, generalised
//! from quads to arbitrary convex face polygons:
//!
//! * The **reference face** is the face (on `A` or `B`) whose outward world
//!   normal is most parallel to the separation. Its owner is the *reference*
//!   hull; the other is the *incident* hull.
//! * The **incident face** is the incident hull's face most anti-parallel to the
//!   reference normal — the face pressing deepest into the reference.
//! * The incident face polygon is clipped against the reference face's side
//!   planes with the Sutherland-Hodgman algorithm. The surviving corners that
//!   lie below the reference face (i.e. actually penetrate) become the manifold
//!   points, each placed on the mid-overlap plane halfway between the incident
//!   corner and the reference face.
//!
//! Unlike the oriented-box path, a convex face is an arbitrary convex polygon
//! and the hull carries any number of faces, so both the reference/incident
//! search and the clip buffers run over dynamic polygons rather than fixed
//! quads. The reduction back to four corners is the shared
//! [`reduce_to_four`](super::manifold::reduce_to_four), so a box fed through
//! this path and the dedicated [`obb_obb_manifold`](super::obb_obb_manifold)
//! path agree point for point.
//!
//! # Normal and depth conventions
//!
//! The manifold's [`normal`](ContactManifold::normal) points from body `a`
//! toward body `b`, matching every other narrow-phase slice and the solver's
//! expectation; it is the negation of the `EPA` push-out normal (which points
//! `B` toward `A`). Each corner's `depth` is its own penetration below the
//! reference face (always positive), and its `position` sits on the
//! mid-overlap plane along the reference normal.
//!
//! # Degenerate fallback
//!
//! When clipping leaves no penetrating corner (a near-grazing contact) or `EPA`
//! cannot build a non-flat polytope (exact tangency), the manifold falls back to
//! the single mid-overlap point between the `EPA` witnesses, so a couple that
//! still overlaps is never silently dropped.
//!
//! Provenance: textbook `GJK`/`EPA` plus reference/incident face-clipping
//! contact manifold; no Unreal Engine source or derived code.

use glam::Vec3;

use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::epa::{epa, Penetration};
use super::gjk::{gjk, GjkStatus};
use super::manifold::{reduce_to_four, ContactManifold, ManifoldPoint};

/// A candidate convex-versus-convex couple: the indices of the two bodies whose
/// manifold should be built.
///
/// Both `a` and `b` index the parallel `hulls` and `poses` slices passed to
/// [`cpu_convex_convex_manifold`] in lockstep, so body `i` is the hull
/// `hulls[i]` placed at `poses[i]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvexConvexPair {
    /// Index of the first body (the `a` side of the contact).
    pub a: u32,
    /// Index of the second body (the `b` side of the contact).
    pub b: u32,
}

impl ConvexConvexPair {
    /// Builds a couple from the two body indices.
    #[must_use]
    pub fn new(a: u32, b: u32) -> ConvexConvexPair {
        ConvexConvexPair { a, b }
    }
}

/// World outward normal of `hull`'s face `face_idx` under `pose`.
fn face_world_normal(hull: &ConvexHull, pose: &ConvexPose, face_idx: usize) -> Vec3 {
    pose.rotation * hull.faces()[face_idx].normal
}

/// World-space polygon (the loop of face corners) of `hull`'s face `face_idx`
/// under `pose`, preserving the face's counter-clockwise winding.
fn face_polygon_world(hull: &ConvexHull, pose: &ConvexPose, face_idx: usize) -> Vec<Vec3> {
    let verts = hull.vertices();
    hull.faces()[face_idx]
        .loop_indices
        .iter()
        .map(|&vi| pose.transform_point(verts[vi as usize]))
        .collect()
}

/// Index of `hull`'s face whose world outward normal is most parallel to
/// `dir`, with the achieved alignment `dot`.
///
/// Ties resolve to the lower face index so the choice is deterministic.
fn most_aligned_face(hull: &ConvexHull, pose: &ConvexPose, dir: Vec3) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_dot = face_world_normal(hull, pose, 0).dot(dir);
    for i in 1..hull.faces().len() {
        let d = face_world_normal(hull, pose, i).dot(dir);
        if d > best_dot {
            best_dot = d;
            best = i;
        }
    }
    (best, best_dot)
}

/// Clips the convex polygon `poly` against the half-space
/// `dot(v - point, normal) <= 0` with the Sutherland-Hodgman algorithm,
/// returning the (still convex) survivor polygon.
///
/// An inside vertex is kept, and an edge straddling the plane contributes its
/// intersection point, so the output has at most one more vertex than the
/// input.
fn clip_poly_to_halfplane(poly: &[Vec3], point: Vec3, normal: Vec3) -> Vec<Vec3> {
    let n = poly.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out: Vec<Vec3> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let cur = poly[i];
        let next = poly[(i + 1) % n];
        let dc = (cur - point).dot(normal);
        let dn = (next - point).dot(normal);
        let cur_in = dc <= 0.0;
        let next_in = dn <= 0.0;
        if cur_in {
            out.push(cur);
        }
        if cur_in != next_in {
            // The edge straddles the plane: the two distances have opposite
            // signs, so the denominator is non-zero.
            let t = dc / (dc - dn);
            out.push(cur + (next - cur) * t);
        }
    }
    out
}

/// Builds the manifold once `EPA` has reported a penetration.
fn build_from_penetration(
    a_id: u32,
    b_id: u32,
    hull_a: &ConvexHull,
    pose_a: &ConvexPose,
    hull_b: &ConvexHull,
    pose_b: &ConvexPose,
    pen: &Penetration,
) -> ContactManifold {
    // `pen.normal` is the push-out normal from B toward A; the manifold reports
    // the a -> b direction, which is its negation.
    let push = pen.normal;
    let normal_ab = -push;

    // Reference-face search: A's face toward B points along -push; B's face
    // toward A points along +push. The better-aligned face is the reference.
    let (face_a, align_a) = most_aligned_face(hull_a, pose_a, -push);
    let (face_b, align_b) = most_aligned_face(hull_b, pose_b, push);

    // Resolve the reference/incident split. `rn` is the reference face's world
    // outward normal; the incident face is the one most anti-parallel to it.
    let reference_is_a = align_a >= align_b;
    let (ref_hull, ref_pose, ref_face, inc_hull, inc_pose) = if reference_is_a {
        (hull_a, pose_a, face_a, hull_b, pose_b)
    } else {
        (hull_b, pose_b, face_b, hull_a, pose_a)
    };
    let rn = face_world_normal(ref_hull, ref_pose, ref_face);
    let ref_poly = face_polygon_world(ref_hull, ref_pose, ref_face);
    let ref_point = ref_poly[0];

    // Incident face: most anti-parallel to the reference normal.
    let (inc_face, _) = most_aligned_face(inc_hull, inc_pose, -rn);
    let mut poly = face_polygon_world(inc_hull, inc_pose, inc_face);

    // Clip the incident polygon against each side plane of the reference face.
    // For a loop wound counter-clockwise about `rn`, the outward side normal of
    // edge (p_i -> p_{i+1}) is `(p_{i+1} - p_i).cross(rn)`.
    let rn_count = ref_poly.len();
    for i in 0..rn_count {
        if poly.is_empty() {
            break;
        }
        let p0 = ref_poly[i];
        let p1 = ref_poly[(i + 1) % rn_count];
        let side = (p1 - p0).cross(rn);
        poly = clip_poly_to_halfplane(&poly, p0, side);
    }

    // Keep the clipped corners that penetrate the reference face; place each on
    // the mid-overlap plane along the reference normal.
    let mut kept: Vec<ManifoldPoint> = Vec::with_capacity(poly.len());
    for corner in &poly {
        let sep = (*corner - ref_point).dot(rn);
        if sep <= 0.0 {
            let depth = -sep;
            let position = *corner + rn * (depth * 0.5);
            kept.push(ManifoldPoint::new(position, depth));
        }
    }

    if kept.is_empty() {
        return single_point_fallback(a_id, b_id, normal_ab, pen);
    }

    let reduced = reduce_to_four(&kept, normal_ab);
    ContactManifold::new(a_id, b_id, normal_ab, reduced.len(), &reduced)
}

/// Single mid-overlap point between the `EPA` witnesses, used when the clip
/// yields no penetrating corner so the still-overlapping couple is not dropped.
fn single_point_fallback(
    a_id: u32,
    b_id: u32,
    normal_ab: Vec3,
    pen: &Penetration,
) -> ContactManifold {
    let position = (pen.point_a + pen.point_b) * 0.5;
    let point = ManifoldPoint::new(position, pen.depth);
    ContactManifold::new(a_id, b_id, normal_ab, 1, &[point])
}

/// Contact manifold for one convex-versus-convex couple, or [`None`] when the
/// two posed hulls are disjoint.
pub(crate) fn convex_convex_manifold_contact(
    a_id: u32,
    b_id: u32,
    hull_a: &ConvexHull,
    pose_a: &ConvexPose,
    hull_b: &ConvexHull,
    pose_b: &ConvexPose,
) -> Option<ContactManifold> {
    let simplex = match gjk(hull_a, pose_a, hull_b, pose_b) {
        GjkStatus::Separated { .. } => return None,
        GjkStatus::Intersecting(simplex) => simplex,
    };
    let Some(pen) = epa(hull_a, pose_a, hull_b, pose_b, &simplex) else {
        // Exact tangency: EPA cannot build a polytope. Report a zero-depth
        // touching point at the shallow simplex witness so the couple is still
        // visible to the solver.
        let witness = simplex
            .iter()
            .map(|s| (s.on_a + s.on_b) * 0.5)
            .fold(Vec3::ZERO, |acc, p| acc + p)
            / simplex.len() as f32;
        let point = ManifoldPoint::new(witness, 0.0);
        // With no penetration normal, fall back to the GJK seed direction
        // between the hull centres (a -> b).
        let dir = pose_b.translation - pose_a.translation;
        let normal_ab = if dir.length_squared() > 1.0e-20 {
            dir.normalize()
        } else {
            Vec3::X
        };
        return Some(ContactManifold::new(a_id, b_id, normal_ab, 1, &[point]));
    };
    Some(build_from_penetration(
        a_id, b_id, hull_a, pose_a, hull_b, pose_b, &pen,
    ))
}

/// Builds a contact manifold for every candidate convex-versus-convex `pair`,
/// one output slot per input couple in order.
///
/// Body `i` is the hull `hulls[i]` placed at `poses[i]`; a couple references
/// those parallel slices by index. A slot is [`None`] when its two hulls are
/// disjoint. Emitting a slot per pair (rather than compacting) keeps the output
/// aligned with the broad-phase candidate list, slot for slot.
///
/// # Panics
///
/// Panics if a couple references a body index outside `hulls` or `poses`, or if
/// the two slices differ in length so an index is valid for one but not the
/// other.
#[must_use]
pub fn cpu_convex_convex_manifold(
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    pairs: &[ConvexConvexPair],
) -> Vec<Option<ContactManifold>> {
    assert_eq!(
        hulls.len(),
        poses.len(),
        "hull and pose slices must align one body per index"
    );
    pairs
        .iter()
        .map(|pair| {
            let a = pair.a as usize;
            let b = pair.b as usize;
            convex_convex_manifold_contact(
                pair.a, pair.b, &hulls[a], &poses[a], &hulls[b], &poses[b],
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrowphase::obb::Obb;
    use crate::narrowphase::obb_obb_manifold::obb_obb_manifold_contact;
    use glam::Quat;

    /// A hull/pose body for a unit-axis box of the given half extents at
    /// `center`.
    fn box_body(center: Vec3, he: Vec3) -> (ConvexHull, ConvexPose) {
        (
            ConvexHull::from_box(he),
            ConvexPose::new(center, Quat::IDENTITY),
        )
    }

    #[test]
    fn face_to_face_overlap_reports_four_coplanar_points() {
        // Box b sits on box a, overlapping 0.5 in y over the full x/z square.
        let (ha, pa) = box_body(Vec3::ZERO, Vec3::ONE);
        let (hb, pb) = box_body(Vec3::new(0.0, 1.5, 0.0), Vec3::ONE);
        let hulls = [ha, hb];
        let poses = [pa, pb];
        let pairs = [ConvexConvexPair::new(0, 1)];
        let m = cpu_convex_convex_manifold(&hulls, &poses, &pairs)[0]
            .expect("overlapping boxes form a manifold");

        assert_eq!(m.count, 4, "a flat face stack has four corners");
        // Normal runs a -> b, i.e. +y here.
        assert!((m.normal - Vec3::Y).length() < 1.0e-4, "normal {:?}", m.normal);
        assert!((m.normal.length() - 1.0).abs() < 1.0e-5);
        for p in m.points.iter().take(m.count as usize) {
            assert!((p.depth - 0.5).abs() < 1.0e-4, "depth {}", p.depth);
            // Mid-overlap plane sits at y = 1.0 - 0.25 = 0.75.
            assert!((p.position.y - 0.75).abs() < 1.0e-4, "y {}", p.position.y);
        }
    }

    #[test]
    fn disjoint_boxes_report_no_manifold() {
        let (ha, pa) = box_body(Vec3::ZERO, Vec3::ONE);
        let (hb, pb) = box_body(Vec3::new(0.0, 5.0, 0.0), Vec3::ONE);
        let hulls = [ha, hb];
        let poses = [pa, pb];
        let pairs = [ConvexConvexPair::new(0, 1)];
        assert!(cpu_convex_convex_manifold(&hulls, &poses, &pairs)[0].is_none());
    }

    #[test]
    fn rotated_box_stack_has_unit_normal_and_bounded_points() {
        // Box b yawed 30 degrees, lowered so it overlaps box a's top face.
        let (ha, pa) = box_body(Vec3::ZERO, Vec3::ONE);
        let hb = ConvexHull::from_box(Vec3::ONE);
        let pb = ConvexPose::new(
            Vec3::new(0.0, 1.5, 0.0),
            Quat::from_rotation_y(core::f32::consts::FRAC_PI_6),
        );
        let hulls = [ha, hb];
        let poses = [pa, pb];
        let pairs = [ConvexConvexPair::new(0, 1)];
        let m = cpu_convex_convex_manifold(&hulls, &poses, &pairs)[0]
            .expect("overlapping yawed boxes form a manifold");

        assert!((m.normal.length() - 1.0).abs() < 1.0e-5);
        assert!((m.normal - Vec3::Y).length() < 1.0e-3, "normal {:?}", m.normal);
        assert!(
            (1..=4).contains(&m.count),
            "point count {} out of range",
            m.count
        );
        for p in m.points.iter().take(m.count as usize) {
            assert!(p.depth > 0.0, "live point must penetrate, got {}", p.depth);
        }
    }

    #[test]
    fn matches_dedicated_obb_path_for_a_box_pair() {
        // The same two boxes fed through the ConvexHull path and the dedicated
        // Obb path must agree on normal and depth set.
        let center_b = Vec3::new(0.3, 1.6, -0.2);
        let he = Vec3::new(1.0, 1.0, 1.0);

        let hulls = [ConvexHull::from_box(he), ConvexHull::from_box(he)];
        let poses = [
            ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
            ConvexPose::new(center_b, Quat::IDENTITY),
        ];
        let pairs = [ConvexConvexPair::new(0, 1)];
        let convex = cpu_convex_convex_manifold(&hulls, &poses, &pairs)[0]
            .expect("convex path manifold");

        let a = Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], he);
        let b = Obb::new(center_b, [Vec3::X, Vec3::Y, Vec3::Z], he);
        let obb = obb_obb_manifold_contact(0, 1, &a, &b).expect("obb path manifold");

        assert_eq!(convex.count, obb.count, "same corner count");
        assert!(
            (convex.normal - obb.normal).length() < 1.0e-3,
            "normals disagree: convex {:?} obb {:?}",
            convex.normal,
            obb.normal
        );
        // Depths form the same multiset (clip order may differ).
        let mut cd: Vec<f32> = convex.points.iter().take(convex.count as usize).map(|p| p.depth).collect();
        let mut od: Vec<f32> = obb.points.iter().take(obb.count as usize).map(|p| p.depth).collect();
        cd.sort_by(|x, y| x.partial_cmp(y).unwrap());
        od.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (c, o) in cd.iter().zip(od.iter()) {
            assert!((c - o).abs() < 1.0e-3, "depth mismatch {c} vs {o}");
        }
    }
}
