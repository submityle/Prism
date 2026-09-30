//! Multi-point OBB-versus-OBB contact manifolds via reference-face clipping.
//!
//! [`obb_obb_contact`](super::obb_obb::obb_obb_contact) reports one
//! representative point per couple, which is enough to un-penetrate two boxes
//! but not to hold a face flush against a face without rocking. This module
//! promotes that single point to a full manifold: up to [`MAX_MANIFOLD_POINTS`]
//! coplanar contact corners sharing the separating-axis normal, the contact a
//! stacking-stable solver consumes.
//!
//! # From separating axis to a manifold
//!
//! The shared [`obb_obb_sat`](super::obb_obb::obb_obb_sat) returns the
//! minimum-translation axis, its oriented normal (`a` toward `b`), and the
//! penetration depth. The winning axis decides the manifold's shape:
//!
//! * **Face contact** (axis is a face normal of `a` or `b`): the owning box is
//!   the *reference* and its winning face the *reference face*; the other box is
//!   the *incident* box and its most anti-parallel face the *incident face*. The
//!   incident face (a quad) is clipped against the four side planes of the
//!   reference face with the Sutherland-Hodgman algorithm, then the surviving
//!   corners that lie below the reference face (i.e. actually penetrate) become
//!   the manifold points. This is the classic reference/incident clip used by
//!   `Box2D`, Bullet, and Havok.
//! * **Edge-edge contact** (axis is an edge-edge cross): the deepest features
//!   are two skew edges, so the manifold degenerates to the single closest-point
//!   pair between them — one point is the correct, complete answer.
//!
//! # Normal and depth conventions
//!
//! The manifold's [`normal`](ContactManifold::normal) is the separating-axis
//! normal, oriented from box `a` toward box `b`, identical to the single-point
//! path. Each corner's `depth` is its own penetration below the reference face
//! (always positive), and its `position` sits on the mid-overlap plane
//! (halfway between the incident corner and the reference face along the
//! reference normal), matching the mid-overlap convention of every other
//! narrow-phase slice.
//!
//! # Point reduction
//!
//! Clipping a quad against four planes can leave more than four corners. When it
//! does, the manifold is reduced to the four that best bound the contact
//! polygon: the deepest corner, the corner farthest from it, and the two corners
//! that maximise the signed area to either side of that diagonal. This is the
//! standard four-point reduction; it keeps the widest, deepest quad so the
//! solver resists both translation and rotation.
//!
//! Provenance: textbook reference/incident face-clipping contact manifold; no
//! Unreal Engine source or derived code.

use glam::Vec3;

use super::manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};
use super::obb::Obb;
use super::obb_obb::{obb_obb_sat, ObbObbPair};

/// Upper bound on corners the Sutherland-Hodgman clip can produce: a quad
/// clipped by four half-planes gains at most one vertex per plane.
const MAX_CLIP_POINTS: usize = 8;

/// Returns `+1.0` when `x >= 0.0`, otherwise `-1.0`; matches the shared
/// support-vertex sign convention so the edge-edge fallback picks the same
/// features as the single-point path.
fn sign_pos(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// The four world vertices of the box face perpendicular to local axis
/// `axis`, on the `sign` side, wound as a loop.
///
/// `u` and `v` are the other two local axis indices (in ascending order) whose
/// half extents span the face; winding the corners `(+u+v, -u+v, -u-v, +u-v)`
/// keeps the polygon a simple loop for clipping.
fn face_quad(b: &Obb, axis: usize, sign: f32) -> [Vec3; 4] {
    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;
    let center = b.center + b.axes[axis] * (sign * b.half_extents[axis]);
    let du = b.axes[u] * b.half_extents[u];
    let dv = b.axes[v] * b.half_extents[v];
    [
        center + du + dv,
        center - du + dv,
        center - du - dv,
        center + du - dv,
    ]
}

/// Chooses the incident face of `b`: the local axis and sign whose outward
/// normal is most anti-parallel to the reference normal `rn`.
///
/// The face most opposed to `rn` is the one that will overlap the reference
/// face, so its four corners form the incident polygon to clip.
fn incident_face(b: &Obb, rn: Vec3) -> (usize, f32) {
    let mut axis = 0usize;
    let mut best = rn.dot(b.axes[0]).abs();
    for (i, a) in b.axes.iter().enumerate().skip(1) {
        let d = rn.dot(*a).abs();
        if d > best {
            best = d;
            axis = i;
        }
    }
    // The incident face normal must oppose rn: pick the sign that makes
    // dot(axis * sign, rn) negative.
    let sign = -sign_pos(rn.dot(b.axes[axis]));
    (axis, sign)
}

/// Clips convex polygon `poly` (its first `n` vertices) against the half-space
/// `dot(v - point, normal) <= 0`, writing survivors into `out` and returning the
/// new count.
///
/// Standard Sutherland-Hodgman: an inside vertex is kept, and an edge crossing
/// the plane contributes its intersection point. The result is convex and has at
/// most one more vertex than the input per clip.
fn clip_to_plane(
    poly: &[Vec3; MAX_CLIP_POINTS],
    n: usize,
    point: Vec3,
    normal: Vec3,
    out: &mut [Vec3; MAX_CLIP_POINTS],
) -> usize {
    let mut count = 0usize;
    for i in 0..n {
        let cur = poly[i];
        let next = poly[(i + 1) % n];
        let dc = (cur - point).dot(normal);
        let dn = (next - point).dot(normal);
        let cur_in = dc <= 0.0;
        let next_in = dn <= 0.0;
        if cur_in && count < MAX_CLIP_POINTS {
            out[count] = cur;
            count += 1;
        }
        // Edge straddles the plane: emit the intersection.
        if (cur_in && !next_in) || (!cur_in && next_in) {
            let denom = dc - dn;
            // denom is non-zero here: the two distances have opposite signs.
            let t = dc / denom;
            if count < MAX_CLIP_POINTS {
                out[count] = cur + (next - cur) * t;
                count += 1;
            }
        }
    }
    count
}

/// Closest points between segments `[p1, q1]` and `[p2, q2]`.
///
/// Ericson's clamped parametric solver: it finds the parameters minimising the
/// squared distance between the two segments, clamping to the endpoints when the
/// unconstrained minimum falls outside `[0, 1]`. Used for the edge-edge contact
/// point.
fn closest_points_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    // Degenerate segments collapse to their start points.
    let eps = 1.0e-12;
    let (mut s, mut t);
    if a <= eps && e <= eps {
        return (p1, p2);
    }
    if a <= eps {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= eps {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            s = if denom > eps {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    (p1 + d1 * s, p2 + d2 * t)
}

/// Reduces more than four contact corners to the four that best bound the
/// contact polygon.
///
/// Keeps the deepest corner, the corner farthest from it, and the two corners
/// that maximise the signed area to either side of that diagonal (measured about
/// `normal`). Preserves the widest, deepest quad so the solver resists both
/// sliding and rocking.
fn reduce_points(
    points: &[ManifoldPoint],
    normal: Vec3,
) -> ([ManifoldPoint; MAX_MANIFOLD_POINTS], usize) {
    let mut out = [ManifoldPoint::new(Vec3::ZERO, 0.0); MAX_MANIFOLD_POINTS];
    if points.len() <= MAX_MANIFOLD_POINTS {
        for (slot, p) in out.iter_mut().zip(points.iter()) {
            *slot = *p;
        }
        return (out, points.len());
    }

    // Deepest corner anchors the quad.
    let mut i0 = 0usize;
    for (i, p) in points.iter().enumerate() {
        if p.depth > points[i0].depth {
            i0 = i;
        }
    }
    // Farthest corner from the anchor forms the diagonal.
    let mut i1 = 0usize;
    let mut best = -1.0;
    for (i, p) in points.iter().enumerate() {
        let d = (p.position - points[i0].position).length_squared();
        if d > best {
            best = d;
            i1 = i;
        }
    }
    // The two corners maximising the signed area either side of the diagonal.
    let edge = points[i1].position - points[i0].position;
    let mut i2 = i0;
    let mut i3 = i0;
    let mut max_area = 0.0;
    let mut min_area = 0.0;
    for (i, p) in points.iter().enumerate() {
        let area = normal.dot(edge.cross(p.position - points[i0].position));
        if area > max_area {
            max_area = area;
            i2 = i;
        }
        if area < min_area {
            min_area = area;
            i3 = i;
        }
    }

    let chosen = [i0, i1, i2, i3];
    let mut count = 0usize;
    for &idx in &chosen {
        // Skip a repeated index (a degenerate reduction may reuse the anchor).
        if chosen[..count].contains(&idx) {
            continue;
        }
        out[count] = points[idx];
        count += 1;
    }
    (out, count)
}

/// Builds the multi-point manifold for one penetrating couple, or [`None`] when
/// the boxes are separated.
#[must_use]
pub(crate) fn obb_obb_manifold_contact(
    a_id: u32,
    b_id: u32,
    a: &Obb,
    b: &Obb,
) -> Option<ContactManifold> {
    let sat = obb_obb_sat(a, b)?;
    let normal = sat.normal;

    if sat.axis_index >= 6 {
        return Some(edge_edge_manifold(a_id, b_id, a, b, &sat));
    }

    // Face contact: identify the reference box/face and the incident face.
    let (reference, incident, rn) = if sat.axis_index < 3 {
        // The winning face belongs to box a; rn points a -> b.
        (a, b, normal)
    } else {
        // The winning face belongs to box b; rn points b -> a.
        (b, a, -normal)
    };
    let ref_axis = if sat.axis_index < 3 {
        sat.axis_index
    } else {
        sat.axis_index - 3
    };
    let ref_sign = sign_pos(rn.dot(reference.axes[ref_axis]));
    let ref_center =
        reference.center + reference.axes[ref_axis] * (ref_sign * reference.half_extents[ref_axis]);

    let (inc_axis, inc_sign) = incident_face(incident, rn);
    let inc_quad = face_quad(incident, inc_axis, inc_sign);

    // Clip the incident quad against the reference face's four side planes.
    let ru = (ref_axis + 1) % 3;
    let rv = (ref_axis + 2) % 3;
    let u = reference.axes[ru];
    let v = reference.axes[rv];
    let hu = reference.half_extents[ru];
    let hv = reference.half_extents[rv];
    let side_planes = [
        (ref_center + u * hu, u),
        (ref_center - u * hu, -u),
        (ref_center + v * hv, v),
        (ref_center - v * hv, -v),
    ];

    let mut buf_a = [Vec3::ZERO; MAX_CLIP_POINTS];
    let mut buf_b = [Vec3::ZERO; MAX_CLIP_POINTS];
    for (i, corner) in inc_quad.iter().enumerate() {
        buf_a[i] = *corner;
    }
    let mut n = inc_quad.len();
    let mut from_a = true;
    for (point, plane_normal) in side_planes {
        let (src, dst): (&[Vec3; MAX_CLIP_POINTS], &mut [Vec3; MAX_CLIP_POINTS]) = if from_a {
            (&buf_a, &mut buf_b)
        } else {
            (&buf_b, &mut buf_a)
        };
        n = clip_to_plane(src, n, point, plane_normal, dst);
        from_a = !from_a;
        if n == 0 {
            break;
        }
    }
    let clipped = if from_a { &buf_a } else { &buf_b };

    // Keep the clipped corners that penetrate the reference face; place each on
    // the mid-overlap plane along the reference normal.
    let mut kept: Vec<ManifoldPoint> = Vec::with_capacity(MAX_CLIP_POINTS);
    for corner in clipped.iter().take(n) {
        let sep = (*corner - ref_center).dot(rn);
        if sep <= 0.0 {
            let depth = -sep;
            let position = *corner + rn * (depth * 0.5);
            kept.push(ManifoldPoint::new(position, depth));
        }
    }

    if kept.is_empty() {
        // Numerical fallback: no corner survived the penetration test (a
        // near-grazing face). Report the single mid-overlap point so the couple
        // is never silently dropped while it still penetrates.
        return Some(single_point_fallback(a_id, b_id, a, b, &sat));
    }

    let (points, count) = reduce_points(&kept, normal);
    Some(ContactManifold::new(a_id, b_id, normal, count, &points))
}

/// Single-point manifold for an edge-edge contact.
fn edge_edge_manifold(
    a_id: u32,
    b_id: u32,
    a: &Obb,
    b: &Obb,
    sat: &super::obb_obb::SatQuery,
) -> ContactManifold {
    let i = (sat.axis_index - 6) / 3;
    let j = (sat.axis_index - 6) % 3;
    let normal = sat.normal;

    // The edge of a whose two off-axis extents lean toward b along +normal.
    let (ai0, ai1) = ((i + 1) % 3, (i + 2) % 3);
    let base_a = a.center
        + a.axes[ai0] * (sign_pos(normal.dot(a.axes[ai0])) * a.half_extents[ai0])
        + a.axes[ai1] * (sign_pos(normal.dot(a.axes[ai1])) * a.half_extents[ai1]);
    let ea = a.axes[i] * a.half_extents[i];
    let pa0 = base_a - ea;
    let pa1 = base_a + ea;

    // The edge of b leaning toward a along -normal.
    let (bj0, bj1) = ((j + 1) % 3, (j + 2) % 3);
    let base_b = b.center
        + b.axes[bj0] * (sign_pos((-normal).dot(b.axes[bj0])) * b.half_extents[bj0])
        + b.axes[bj1] * (sign_pos((-normal).dot(b.axes[bj1])) * b.half_extents[bj1]);
    let eb = b.axes[j] * b.half_extents[j];
    let pb0 = base_b - eb;
    let pb1 = base_b + eb;

    let (ca, cb) = closest_points_segments(pa0, pa1, pb0, pb1);
    let point = (ca + cb) * 0.5;
    ContactManifold::new(
        a_id,
        b_id,
        normal,
        1,
        &[ManifoldPoint::new(point, sat.depth)],
    )
}

/// Mid-overlap single point, mirroring the single-point narrow phase, used only
/// as a numerical fallback when face clipping keeps no corner.
fn single_point_fallback(
    a_id: u32,
    b_id: u32,
    a: &Obb,
    b: &Obb,
    sat: &super::obb_obb::SatQuery,
) -> ContactManifold {
    let normal = sat.normal;
    let pa = support_vertex(a, normal);
    let pb = support_vertex(b, -normal);
    let point = (pa + pb) * 0.5;
    ContactManifold::new(
        a_id,
        b_id,
        normal,
        1,
        &[ManifoldPoint::new(point, sat.depth)],
    )
}

/// Support vertex of a box in direction `dir`: the corner furthest along `dir`.
fn support_vertex(b: &Obb, dir: Vec3) -> Vec3 {
    b.center
        + b.axes[0] * (sign_pos(dir.dot(b.axes[0])) * b.half_extents.x)
        + b.axes[1] * (sign_pos(dir.dot(b.axes[1])) * b.half_extents.y)
        + b.axes[2] * (sign_pos(dir.dot(b.axes[2])) * b.half_extents.z)
}

/// `CPU` golden twin of the multi-point OBB-OBB narrow phase.
///
/// Turns candidate `pairs` into contact manifolds, one slot per couple in input
/// order: [`Some`] carrying up to four contact points when the boxes penetrate,
/// or [`None`] when a separating axis exists. Mirrors
/// [`cpu_obb_obb_narrowphase`](super::obb_obb::cpu_obb_obb_narrowphase) but with
/// the richer manifold the solver needs for stable stacking.
///
/// # Panics
///
/// Panics if a couple references a box index outside `boxes`.
#[must_use]
pub fn cpu_obb_obb_manifold(boxes: &[Obb], pairs: &[ObbObbPair]) -> Vec<Option<ContactManifold>> {
    pairs
        .iter()
        .map(|pair| {
            let a = &boxes[pair.a as usize];
            let b = &boxes[pair.b as usize];
            obb_obb_manifold_contact(pair.a, pair.b, a, b)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// A unit-axis box at `center` with the given half extents.
    fn axis_box(center: Vec3, he: Vec3) -> Obb {
        Obb::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
    }

    #[test]
    fn flat_stack_reports_four_coplanar_points() {
        // Box b resting on box a, overlapping 0.5 in y over the full x/z square.
        // The manifold must be the four corners of that square, all at depth 0.5.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(0.0, 1.5, 0.0), Vec3::ONE);
        let m = obb_obb_manifold_contact(0, 1, &a, &b).expect("stacked boxes overlap");
        assert_eq!(m.a, 0);
        assert_eq!(m.b, 1);
        assert_eq!(m.count, 4);
        assert!((m.normal - Vec3::Y).length() < 1.0e-6);
        for point in m.points.iter().take(m.count as usize) {
            assert!((point.depth - 0.5).abs() < 1.0e-5, "depth {}", point.depth);
            // Each corner sits on the mid-overlap plane y = 0.75.
            assert!((point.position.y - 0.75).abs() < 1.0e-5);
            assert!(point.position.x.abs() <= 1.0 + 1.0e-5);
            assert!(point.position.z.abs() <= 1.0 + 1.0e-5);
        }
    }

    #[test]
    fn offset_stack_clips_to_the_overlap_column() {
        // Shift b in +x so its face overhangs a: clipping must trim the manifold
        // to the shared column x in [-0.5, 1].
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(0.5, 1.5, 0.0), Vec3::ONE);
        let m = obb_obb_manifold_contact(0, 1, &a, &b).expect("boxes overlap");
        assert_eq!(m.count, 4);
        let max_x = m
            .points
            .iter()
            .take(m.count as usize)
            .map(|p| p.position.x)
            .fold(f32::MIN, f32::max);
        let min_x = m
            .points
            .iter()
            .take(m.count as usize)
            .map(|p| p.position.x)
            .fold(f32::MAX, f32::min);
        assert!((max_x - 1.0).abs() < 1.0e-5, "max_x {max_x}");
        assert!((min_x + 0.5).abs() < 1.0e-5, "min_x {min_x}");
    }

    #[test]
    fn rotated_face_overlap_reduces_to_four_points() {
        // A box yawed 45 degrees resting on an axis-aligned box: the incident
        // diamond clipped against the reference square is an octagon, which the
        // reduction must trim to four points.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let yaw = Quat::from_rotation_y(core::f32::consts::FRAC_PI_4);
        let b = Obb::from_quat(Vec3::new(0.0, 1.5, 0.0), yaw, Vec3::ONE);
        let m = obb_obb_manifold_contact(0, 1, &a, &b).expect("boxes overlap");
        assert!((m.normal - Vec3::Y).length() < 1.0e-6);
        assert_eq!(m.count, 4, "octagon overlap reduces to four points");
        for point in m.points.iter().take(m.count as usize) {
            assert!(point.depth > 0.0);
        }
    }

    #[test]
    fn edge_edge_contact_is_a_single_point() {
        // The edge-edge configuration from the single-point slice must yield one
        // contact point, not a face manifold.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let rot = Quat::from_euler(
            glam::EulerRot::XYZ,
            core::f32::consts::FRAC_PI_4,
            0.0,
            core::f32::consts::FRAC_PI_4,
        );
        let b = Obb::from_quat(Vec3::new(1.6, 1.6, 0.0), rot, Vec3::ONE);
        let m = obb_obb_manifold_contact(0, 1, &a, &b).expect("edge overlap");
        assert_eq!(m.count, 1);
        assert!((m.normal.length() - 1.0).abs() < 1.0e-5);
        assert!(m.points[0].depth > 0.0);
    }

    #[test]
    fn separated_boxes_report_no_manifold() {
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(10.0, 0.0, 0.0), Vec3::ONE);
        assert!(obb_obb_manifold_contact(0, 1, &a, &b).is_none());
    }

    #[test]
    fn manifold_normal_matches_single_point_path() {
        // The manifold must share the separating-axis normal with the
        // single-point narrow phase for the same couple.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(0.3, 1.5, 0.2), Vec3::ONE);
        let single = super::super::obb_obb::obb_obb_contact(0, 1, &a, &b).expect("boxes overlap");
        let m = obb_obb_manifold_contact(0, 1, &a, &b).expect("boxes overlap");
        assert!((m.normal - single.normal).length() < 1.0e-6);
    }

    #[test]
    fn batch_preserves_order_and_indices() {
        let boxes = [
            axis_box(Vec3::ZERO, Vec3::ONE),
            axis_box(Vec3::new(0.0, 1.5, 0.0), Vec3::ONE),
            axis_box(Vec3::new(20.0, 0.0, 0.0), Vec3::ONE),
        ];
        let pairs = [ObbObbPair::new(0, 1), ObbObbPair::new(0, 2)];
        let out = cpu_obb_obb_manifold(&boxes, &pairs);
        assert_eq!(out.len(), 2);
        let first = out[0].expect("stacked boxes overlap");
        assert_eq!(first.a, 0);
        assert_eq!(first.b, 1);
        assert!(first.count >= 1);
        assert!(out[1].is_none());
    }
}
