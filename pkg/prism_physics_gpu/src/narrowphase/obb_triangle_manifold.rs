//! Multi-point OBB-versus-triangle contact manifolds via reference-face
//! clipping.
//!
//! [`obb_triangle_contact`](super::obb_triangle::obb_triangle_contact) reports
//! one representative point per couple, which un-penetrates a box from a
//! triangle but cannot hold a box face flush on a triangle without rocking.
//! This module promotes that single point to a full manifold: up to
//! `MAX_MANIFOLD_POINTS` coplanar corners sharing the separating-axis normal,
//! the contact a stacking-stable solver consumes. A trimesh or heightfield
//! collider batches these per-triangle manifolds, so a box resting on level
//! geometry gets the four-corner support it needs on every triangle it touches.
//!
//! # From separating axis to a manifold
//!
//! The shared [`obb_triangle_sat`](super::obb_triangle::obb_triangle_sat)
//! returns the minimum-translation axis, its oriented normal (triangle toward
//! box), the penetration depth, and the index of the winning axis in the fixed
//! thirteen-axis order. That index classifies the contact:
//!
//! * **Box-face contact** (axis index `0..=2`, a box face normal): the box face
//!   turned toward the triangle is the *reference face*; the triangle is the
//!   *incident* polygon. The triangle is clipped against the four side planes of
//!   the box's local slab (the `u`/`v` extents spanning the reference face) with
//!   the Sutherland-Hodgman algorithm, and the survivors that lie below the
//!   reference face (i.e. actually inside the box) become the manifold points.
//! * **Triangle-face contact** (axis index `3`, the triangle normal): the
//!   triangle is the *reference face*; the box face most anti-parallel to the
//!   normal is the *incident* face. The incident quad is clipped against the
//!   three edge side planes of the triangle, and the survivors below the
//!   triangle plane become the manifold points.
//! * **Edge-edge contact** (axis index `4..=12`, an edge-edge cross): the
//!   deepest features are two skew edges, so the manifold degenerates to the
//!   single representative closest-point pair — one point is the complete
//!   answer, identical to the single-point path.
//!
//! # Normal and depth conventions
//!
//! The manifold's [`normal`](ContactManifold::normal) is the separating-axis
//! normal oriented **from the triangle (`b`) toward the box (`a`)**, identical
//! to `obb_triangle_contact`, so the trimesh collider sees one consistent
//! convention across the single-point and manifold paths. Each corner's `depth`
//! is its own penetration below the reference face (always positive), and its
//! `position` sits on the mid-overlap plane halfway between the incident corner
//! and the reference face along the normal — the mid-overlap convention every
//! other narrow-phase slice uses.
//!
//! # Point reduction
//!
//! Clipping can leave more than four corners. When it does, the manifold is
//! reduced to the four that best bound the contact polygon: the deepest corner,
//! the corner farthest from it, and the two corners that maximise the signed
//! area to either side of that diagonal. This keeps the widest, deepest quad so
//! the solver resists both translation and rotation — the standard four-point
//! reduction used by `Box2D`, Bullet, and Havok.
//!
//! Provenance: textbook reference/incident face-clipping contact manifold and
//! Sutherland-Hodgman polygon clip (Ericson, *Real-Time Collision Detection*,
//! 2004, sections 5.4 and 8.3). No Unreal Engine source or derived code.

use glam::Vec3;

use super::manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};
use super::obb::Obb;
use super::obb_triangle::{obb_triangle_sat, support_vertex, ObbTrianglePair};
use super::sphere_triangle::{closest_point_on_triangle, Triangle};

/// Upper bound on corners the Sutherland-Hodgman clip can produce: a quad or
/// triangle clipped by four half-planes gains at most one vertex per plane, so
/// a four-vertex incident polygon can reach eight corners before reduction.
const MAX_CLIP_POINTS: usize = 8;

/// Penetration below which a clipped corner is dropped. A corner exactly on the
/// reference plane (`pen == 0`) is a grazing coplanar edge, not a live contact,
/// so the strict `pen > 0` filter matches the single-point path's strict
/// `overlap > 0` rejection.
const PEN_EPS: f32 = 0.0;

/// Returns `+1.0` when `x >= 0.0`, otherwise `-1.0`; matches the shared
/// support-vertex sign convention in `super::obb_triangle`.
fn sign_pos(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Clips a convex polygon against one half-space, keeping the portion on the
/// inward side of the plane `(point, inward)`.
///
/// A vertex is kept when `(v - point) . inward >= 0`; each edge that crosses the
/// plane contributes the interpolated crossing point. The input is treated as a
/// closed loop. Returns the (possibly empty) clipped loop.
fn clip_halfspace(poly: &[Vec3], point: Vec3, inward: Vec3) -> Vec<Vec3> {
    let n = poly.len();
    let mut out: Vec<Vec3> = Vec::with_capacity(n + 1);
    if n == 0 {
        return out;
    }
    for i in 0..n {
        let cur = poly[i];
        let nxt = poly[(i + 1) % n];
        let dc = (cur - point).dot(inward);
        let dn = (nxt - point).dot(inward);
        let cur_in = dc >= 0.0;
        let nxt_in = dn >= 0.0;
        if cur_in {
            out.push(cur);
        }
        // Edge crosses the plane: add the intersection point.
        if cur_in != nxt_in {
            let denom = dc - dn;
            // dc != dn whenever the inside flags differ, so denom is non-zero.
            let t = dc / denom;
            out.push(cur + (nxt - cur) * t);
        }
    }
    out
}

/// Clips `poly` against every half-space in `planes` (each `(point, inward)`),
/// returning the surviving convex loop.
fn clip_against_planes(poly: &[Vec3], planes: &[(Vec3, Vec3)]) -> Vec<Vec3> {
    let mut cur: Vec<Vec3> = poly.to_vec();
    for &(point, inward) in planes {
        if cur.is_empty() {
            break;
        }
        cur = clip_halfspace(&cur, point, inward);
    }
    cur
}

/// The four world corners of the box face perpendicular to local `axis`, on the
/// `sign` side, wound as a loop. `u` and `v` are the other two local axes.
fn box_face_quad(obb: &Obb, axis: usize, sign: f32) -> [Vec3; 4] {
    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;
    let center = obb.center + obb.axes[axis] * (sign * obb.half_extents[axis]);
    let du = obb.axes[u] * obb.half_extents[u];
    let dv = obb.axes[v] * obb.half_extents[v];
    [
        center + du + dv,
        center - du + dv,
        center - du - dv,
        center + du - dv,
    ]
}

/// The four side half-spaces of the box's local slab spanned by local axes `u`
/// and `v`, each `(point, inward)` with the inward normal pointing into the box.
fn box_slab_planes(obb: &Obb, u: usize, v: usize) -> [(Vec3, Vec3); 4] {
    let au = obb.axes[u];
    let av = obb.axes[v];
    let hu = obb.half_extents[u];
    let hv = obb.half_extents[v];
    [
        (obb.center + au * hu, -au),
        (obb.center - au * hu, au),
        (obb.center + av * hv, -av),
        (obb.center - av * hv, av),
    ]
}

/// The three edge side half-spaces of the triangle, each `(point, inward)` with
/// the inward normal lying in the triangle plane and pointing toward the
/// interior (the opposite vertex). `plane_normal` is any vector along the
/// triangle normal; its sign does not matter because each inward normal is
/// re-oriented toward the third vertex.
fn triangle_edge_planes(tri: &Triangle, plane_normal: Vec3) -> [(Vec3, Vec3); 3] {
    let edges = [
        (tri.a, tri.b, tri.c),
        (tri.b, tri.c, tri.a),
        (tri.c, tri.a, tri.b),
    ];
    let mut planes = [(Vec3::ZERO, Vec3::ZERO); 3];
    for (slot, (va, vb, third)) in planes.iter_mut().zip(edges) {
        let mut inward = (vb - va).cross(plane_normal);
        if inward.dot(third - va) < 0.0 {
            inward = -inward;
        }
        *slot = (va, inward);
    }
    planes
}

/// Projects clipped incident corners onto the mid-overlap plane and keeps the
/// penetrating ones.
///
/// `ref_point` is any point on the reference plane and `ref_out` is the
/// reference face's outward unit normal (pointing away from the reference body
/// toward the incident one). A corner `p` penetrates when it lies on the
/// interior side, `pen = (ref_point - p) . ref_out > 0`; its mid-overlap
/// position is `p + ref_out * (pen / 2)`.
fn penetrating_points(clipped: &[Vec3], ref_point: Vec3, ref_out: Vec3) -> Vec<ManifoldPoint> {
    let mut out = Vec::with_capacity(clipped.len());
    for &p in clipped {
        let pen = (ref_point - p).dot(ref_out);
        if pen > PEN_EPS {
            out.push(ManifoldPoint::new(p + ref_out * (pen * 0.5), pen));
        }
    }
    out
}

/// Reduces a set of coplanar manifold points to at most
/// `MAX_MANIFOLD_POINTS`, keeping the widest, deepest quad.
///
/// Picks the deepest corner, the corner farthest from it, then the two corners
/// maximising the signed triangle area to either side of that diagonal
/// (measured in the plane whose normal is `normal`). Fewer than five input
/// points are returned unchanged (order preserved).
fn reduce_points(points: &[ManifoldPoint], normal: Vec3) -> Vec<ManifoldPoint> {
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

/// Builds the single representative point used for the edge-edge contact and as
/// the fallback when clipping leaves no penetrating corner.
fn representative_point(obb: &Obb, tri: &Triangle, normal: Vec3, depth: f32) -> ManifoldPoint {
    let box_point = support_vertex(obb.center, &obb.axes, obb.half_extents, -normal);
    let tri_point = closest_point_on_triangle(box_point, tri.a, tri.b, tri.c);
    ManifoldPoint::new((box_point + tri_point) * 0.5, depth)
}

/// Builds the box-face-reference manifold points (triangle clipped against the
/// box slab).
fn box_face_points(obb: &Obb, tri: &Triangle, normal: Vec3, axis: usize) -> Vec<ManifoldPoint> {
    // Reference face outward normal points from the box toward the triangle,
    // i.e. opposite the contact normal (which runs triangle -> box).
    let face_out_axis = obb.axes[axis];
    let sign = sign_pos((-normal).dot(face_out_axis));
    let ref_out = face_out_axis * sign;
    let ref_point = obb.center + ref_out * obb.half_extents[axis];
    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;
    let tri_poly = [tri.a, tri.b, tri.c];
    let clipped = clip_against_planes(&tri_poly, &box_slab_planes(obb, u, v));
    penetrating_points(&clipped, ref_point, ref_out)
}

/// Builds the triangle-face-reference manifold points (box incident face
/// clipped against the triangle edges).
fn triangle_face_points(obb: &Obb, tri: &Triangle, normal: Vec3) -> Vec<ManifoldPoint> {
    // The triangle is the reference; its outward normal points toward the box,
    // i.e. along the contact normal.
    let ref_out = normal;
    let ref_point = tri.a;
    // Incident box face: the one whose outward normal is most anti-parallel to
    // the contact normal (the face turned toward the triangle).
    let mut axis = 0usize;
    let mut best = -1.0f32;
    for (i, a) in obb.axes.iter().enumerate() {
        let d = a.dot(normal).abs();
        if d > best {
            best = d;
            axis = i;
        }
    }
    let sign = -sign_pos(obb.axes[axis].dot(normal));
    let quad = box_face_quad(obb, axis, sign);
    let clipped = clip_against_planes(&quad, &triangle_edge_planes(tri, ref_out));
    penetrating_points(&clipped, ref_point, ref_out)
}

/// Builds the full contact manifold for a penetrating `(box, triangle)` couple.
///
/// Dispatches on the winning separating axis: a box-face or triangle-face
/// contact is clipped into up to four coplanar corners, while an edge-edge
/// contact yields the single representative point. When clipping leaves no
/// penetrating corner (a numerical corner case), the representative point is
/// used so a reported manifold always carries at least one live point.
#[must_use]
pub(crate) fn obb_triangle_manifold(
    box_id: u32,
    tri_id: u32,
    obb: &Obb,
    tri: &Triangle,
) -> Option<ContactManifold> {
    let sat = obb_triangle_sat(obb, tri)?;
    let normal = sat.normal;

    let mut points = if sat.axis_index <= 2 {
        box_face_points(obb, tri, normal, sat.axis_index)
    } else if sat.axis_index == 3 {
        triangle_face_points(obb, tri, normal)
    } else {
        Vec::new()
    };

    if points.is_empty() {
        points.push(representative_point(obb, tri, normal, sat.depth));
    }

    let reduced = reduce_points(&points, normal);
    debug_assert!(reduced.len() <= MAX_MANIFOLD_POINTS);
    debug_assert!(reduced.len() <= MAX_CLIP_POINTS);
    Some(ContactManifold::new(
        box_id,
        tri_id,
        normal,
        reduced.len(),
        &reduced,
    ))
}

/// `CPU` golden twin of the OBB-versus-triangle manifold narrow phase.
///
/// Turns candidate `pairs` into contact manifolds, one slot per couple in input
/// order: `Some` carrying the clipped manifold when the box penetrates the
/// triangle, or `None` when a separating axis exists. The ordering lets a
/// parity test line up a device kernel's manifolds index by index, and lets a
/// trimesh collider aggregate the per-triangle manifolds into a per-proxy
/// contact set.
///
/// # Panics
///
/// Panics if a couple references a box or triangle index outside the
/// corresponding slice, which a broad phase over the same sets never produces.
#[must_use]
pub fn cpu_obb_triangle_manifold(
    boxes: &[Obb],
    triangles: &[Triangle],
    pairs: &[ObbTrianglePair],
) -> Vec<Option<ContactManifold>> {
    pairs
        .iter()
        .map(|pair| {
            let obb = &boxes[pair.obb as usize];
            let tri = &triangles[pair.triangle as usize];
            obb_triangle_manifold(pair.obb, pair.triangle, obb, tri)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::obb_triangle::obb_triangle_contact;
    use glam::Vec3;

    const EPS: f32 = 1.0e-4;

    fn axis_aligned(center: Vec3, he: Vec3) -> Obb {
        Obb::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
    }

    fn big_floor() -> Triangle {
        // A large triangle in the z = 0 plane that contains any unit footprint
        // around the origin.
        Triangle::new(
            Vec3::new(-10.0, -10.0, 0.0),
            Vec3::new(10.0, -10.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
        )
    }

    #[test]
    fn separated_box_reports_no_manifold() {
        let obb = axis_aligned(Vec3::new(0.0, 0.0, 5.0), Vec3::splat(1.0));
        let tri = big_floor();
        assert!(obb_triangle_manifold(0, 0, &obb, &tri).is_none());
    }

    #[test]
    fn box_resting_flush_yields_four_corners() {
        // Box bottom face sits 0.1 below the floor plane: a full four-corner
        // face contact.
        let obb = axis_aligned(Vec3::new(0.0, 0.0, 0.9), Vec3::splat(1.0));
        let tri = big_floor();
        let m = obb_triangle_manifold(0, 0, &obb, &tri).expect("penetrating");
        assert_eq!(m.count, 4);
        // Normal runs from the triangle up toward the box.
        assert!((m.normal - Vec3::Z).length() < EPS, "normal {:?}", m.normal);
        // Normal agrees with the single-point representative path.
        let single = obb_triangle_contact(0, 0, &obb, &tri).unwrap();
        assert!((m.normal - single.normal).length() < EPS);
        let mut xy: Vec<(i32, i32)> = Vec::new();
        for pt in &m.points[..m.count as usize] {
            assert!((pt.depth - 0.1).abs() < EPS, "depth {}", pt.depth);
            assert!((pt.position.z + 0.05).abs() < EPS, "z {}", pt.position.z);
            xy.push((pt.position.x.round() as i32, pt.position.y.round() as i32));
        }
        xy.sort_unstable();
        assert_eq!(xy, vec![(-1, -1), (-1, 1), (1, -1), (1, 1)]);
    }

    #[test]
    fn small_triangle_poking_a_box_face_yields_three_points() {
        // A small triangle fully inside the box's y/z slab, penetrating the +x
        // face: the whole triangle survives the clip as three points.
        let obb = axis_aligned(Vec3::ZERO, Vec3::splat(2.0));
        let tri = Triangle::new(
            Vec3::new(1.5, -0.5, -0.5),
            Vec3::new(1.5, 0.5, -0.5),
            Vec3::new(1.5, 0.0, 0.5),
        );
        let m = obb_triangle_manifold(0, 0, &obb, &tri).expect("penetrating");
        assert_eq!(m.count, 3);
        assert!((m.normal - (-Vec3::X)).length() < EPS, "normal {:?}", m.normal);
        for pt in &m.points[..m.count as usize] {
            assert!((pt.depth - 0.5).abs() < EPS, "depth {}", pt.depth);
            // Mid-overlap plane sits halfway between the triangle (x = 1.5) and
            // the +x face (x = 2.0), i.e. x = 1.75.
            assert!((pt.position.x - 1.75).abs() < EPS, "x {}", pt.position.x);
        }
    }

    #[test]
    fn clip_halfspace_cuts_a_crossing_edge() {
        // A unit square clipped by the x >= 0 half-space keeps the right half.
        let square = [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ];
        let out = clip_halfspace(&square, Vec3::ZERO, Vec3::X);
        // Four corners: two originals on the right and two interpolated on x = 0.
        assert_eq!(out.len(), 4);
        for p in &out {
            assert!(p.x >= -EPS, "x {}", p.x);
        }
        assert!(out.iter().any(|p| (p.x).abs() < EPS));
    }

    #[test]
    fn reduce_keeps_four_extreme_corners() {
        // Six coplanar points (a hexagon in z = 0); reduction keeps four that
        // bound it, dropping the two interior-edge midpoints.
        let pts = vec![
            ManifoldPoint::new(Vec3::new(-2.0, 0.0, 0.0), 0.3),
            ManifoldPoint::new(Vec3::new(-1.0, 1.0, 0.0), 0.1),
            ManifoldPoint::new(Vec3::new(1.0, 1.0, 0.0), 0.1),
            ManifoldPoint::new(Vec3::new(2.0, 0.0, 0.0), 0.5),
            ManifoldPoint::new(Vec3::new(1.0, -1.0, 0.0), 0.1),
            ManifoldPoint::new(Vec3::new(-1.0, -1.0, 0.0), 0.1),
        ];
        let reduced = reduce_points(&pts, Vec3::Z);
        assert_eq!(reduced.len(), 4);
        // The deepest corner (2, 0) must survive.
        assert!(reduced.iter().any(|p| (p.position - Vec3::new(2.0, 0.0, 0.0)).length() < EPS));
        // The farthest corner from it (-2, 0) must survive.
        assert!(reduced.iter().any(|p| (p.position - Vec3::new(-2.0, 0.0, 0.0)).length() < EPS));
    }

    #[test]
    fn batch_preserves_order_and_misses() {
        let boxes = vec![axis_aligned(Vec3::new(0.0, 0.0, 0.9), Vec3::splat(1.0))];
        let tris = vec![big_floor()];
        let pairs = vec![
            ObbTrianglePair::new(0, 0),
        ];
        let out = cpu_obb_triangle_manifold(&boxes, &tris, &pairs);
        assert_eq!(out.len(), 1);
        assert!(out[0].is_some());
    }
}
