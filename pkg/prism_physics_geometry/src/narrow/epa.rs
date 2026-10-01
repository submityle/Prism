//! Convex penetration recovery via GJK + the Expanding Polytope Algorithm (EPA).
//!
//! When two convex shapes overlap, [`gjk_contact`] reports the minimum
//! translation that separates them: a unit contact `normal`, the penetration
//! `depth`, and a pair of witness points on each shape's surface. The routine
//! first drives GJK to a tetrahedron that encloses the origin of the Minkowski
//! difference `A (-) B`, then grows that tetrahedron face-by-face toward the
//! origin until the closest boundary face is found.
//!
//! This is a clean-room implementation of the publicly documented GJK/EPA
//! algorithms and contains no Unreal Engine source or derived code.

use alloc::vec::Vec;
use glam::Vec3;

use crate::narrow::minkowski::{support, SupportVertex};
use crate::narrow::support::SupportMap;

/// Maximum GJK simplex refinement iterations.
const GJK_MAX_ITERATIONS: usize = 64;

/// Maximum EPA polytope expansion steps.
const EPA_MAX_ITERATIONS: usize = 96;

/// Squared tolerance treating two Minkowski samples as the same vertex.
const DUPLICATE_EPSILON_SQ: f32 = 1.0e-10;

/// Convergence tolerance for EPA face-distance improvement.
const EPA_TOLERANCE: f32 = 1.0e-3;

/// Faces this close to the origin are treated as degenerate artifacts of an
/// origin-on-boundary starting tetrahedron and are ignored by the fallback.
const DEGENERATE_DISTANCE: f32 = 1.0e-4;

/// A contact manifold point produced when two convex shapes overlap.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Contact {
    /// Unit contact normal pointing from shape `a` toward shape `b`.
    ///
    /// Separating the pair requires translating `a` by `-normal * depth`
    /// (equivalently translating `b` by `+normal * depth`).
    pub normal: Vec3,
    /// Penetration depth along `normal` (always non-negative).
    pub depth: f32,
    /// Witness point on the surface of shape `a`.
    pub point_a: Vec3,
    /// Witness point on the surface of shape `b`.
    pub point_b: Vec3,
}

/// Returns `true` when `a` and `b` lie in the same half-space.
fn same_direction(a: Vec3, b: Vec3) -> bool {
    a.dot(b) > 0.0
}

/// Returns an arbitrary non-zero vector perpendicular to `v`.
fn any_perpendicular(v: Vec3) -> Vec3 {
    let c = v.cross(Vec3::X);
    if c.length_squared() > DUPLICATE_EPSILON_SQ {
        c
    } else {
        v.cross(Vec3::Y)
    }
}

/// Reports the overlap of two convex shapes.
///
/// Returns `Some(contact)` with the minimum-translation [`Contact`] when the
/// shapes penetrate, or `None` when they are separate or only grazing (a
/// boundary touch has no well-defined penetration direction).
pub fn gjk_contact<A: SupportMap, B: SupportMap>(a: &A, b: &B) -> Option<Contact> {
    let simplex = gjk_tetrahedron(a, b)?;
    epa(a, b, simplex)
}

/// Drives GJK until the simplex is a tetrahedron enclosing the origin.
///
/// Returns the four Minkowski vertices on success, or `None` when a separating
/// axis is found (no overlap) or the search stalls on a boundary touch.
fn gjk_tetrahedron<A: SupportMap, B: SupportMap>(a: &A, b: &B) -> Option<[SupportVertex; 4]> {
    // `simplex[0]` is always the most recently added vertex.
    let mut simplex: Vec<SupportVertex> = Vec::with_capacity(4);
    let first = support(a, b, Vec3::X);
    simplex.push(first);
    let mut dir = if first.v.length_squared() > DUPLICATE_EPSILON_SQ {
        -first.v
    } else {
        Vec3::X
    };

    for _ in 0..GJK_MAX_ITERATIONS {
        if dir.length_squared() <= DUPLICATE_EPSILON_SQ {
            // Origin lies on the current simplex: a grazing touch with no
            // definite penetration direction.
            return None;
        }
        let point = support(a, b, dir);
        if point.v.dot(dir) < 0.0 {
            // The farthest point fell short of the origin: shapes are apart.
            return None;
        }
        if simplex
            .iter()
            .any(|q| (q.v - point.v).length_squared() < DUPLICATE_EPSILON_SQ)
        {
            // No further progress toward the origin.
            return None;
        }
        simplex.insert(0, point);
        if next_simplex(&mut simplex, &mut dir) {
            return Some([simplex[0], simplex[1], simplex[2], simplex[3]]);
        }
    }
    None
}

/// Advances the front-ordered simplex toward the origin, returning `true` once
/// a tetrahedron encloses it.
fn next_simplex(simplex: &mut Vec<SupportVertex>, dir: &mut Vec3) -> bool {
    match simplex.len() {
        2 => line_case(simplex, dir),
        3 => triangle_case(simplex, dir),
        4 => tetrahedron_case(simplex, dir),
        _ => false,
    }
}

fn line_case(simplex: &mut Vec<SupportVertex>, dir: &mut Vec3) -> bool {
    let a = simplex[0];
    let b = simplex[1];
    let ab = b.v - a.v;
    let ao = -a.v;
    if same_direction(ab, ao) {
        let mut d = ab.cross(ao).cross(ab);
        if d.length_squared() <= DUPLICATE_EPSILON_SQ {
            // Origin lies on the line: search along an arbitrary perpendicular.
            d = any_perpendicular(ab);
        }
        *dir = d;
    } else {
        simplex.clear();
        simplex.push(a);
        *dir = ao;
    }
    false
}

fn triangle_case(simplex: &mut Vec<SupportVertex>, dir: &mut Vec3) -> bool {
    let a = simplex[0];
    let b = simplex[1];
    let c = simplex[2];
    let ab = b.v - a.v;
    let ac = c.v - a.v;
    let ao = -a.v;
    let abc = ab.cross(ac);

    if same_direction(abc.cross(ac), ao) {
        if same_direction(ac, ao) {
            // Closest to edge ac.
            simplex.clear();
            simplex.push(a);
            simplex.push(c);
            *dir = ac.cross(ao).cross(ac);
        } else {
            // Reduce to edge ab.
            simplex.clear();
            simplex.push(a);
            simplex.push(b);
            return line_case(simplex, dir);
        }
    } else if same_direction(ab.cross(abc), ao) {
        // Reduce to edge ab.
        simplex.clear();
        simplex.push(a);
        simplex.push(b);
        return line_case(simplex, dir);
    } else if same_direction(abc, ao) {
        // Origin above the triangle face.
        *dir = abc;
    } else {
        // Origin below the face: flip winding so the normal points at it.
        simplex.clear();
        simplex.push(a);
        simplex.push(c);
        simplex.push(b);
        *dir = -abc;
    }
    false
}

fn tetrahedron_case(simplex: &mut Vec<SupportVertex>, dir: &mut Vec3) -> bool {
    let a = simplex[0];
    let b = simplex[1];
    let c = simplex[2];
    let d = simplex[3];
    let ab = b.v - a.v;
    let ac = c.v - a.v;
    let ad = d.v - a.v;
    let ao = -a.v;

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    if same_direction(abc, ao) {
        simplex.clear();
        simplex.push(a);
        simplex.push(b);
        simplex.push(c);
        return triangle_case(simplex, dir);
    }
    if same_direction(acd, ao) {
        simplex.clear();
        simplex.push(a);
        simplex.push(c);
        simplex.push(d);
        return triangle_case(simplex, dir);
    }
    if same_direction(adb, ao) {
        simplex.clear();
        simplex.push(a);
        simplex.push(d);
        simplex.push(b);
        return triangle_case(simplex, dir);
    }
    // Origin is on the inner side of all four faces: enclosed.
    true
}

/// A polytope face: three vertex indices with a cached outward unit normal and
/// the distance from the origin to the face plane.
#[derive(Clone, Copy)]
struct Face {
    indices: [usize; 3],
    normal: Vec3,
    distance: f32,
    /// Set when the face cannot be expanded further (its outward support is a
    /// duplicate vertex). Degenerate zero-distance faces created when the
    /// origin lies on the initial tetrahedron boundary end up dead, letting
    /// EPA fall through to the faces that bound the true penetration.
    dead: bool,
}

/// Builds a face from three vertices, orienting the normal to point away from
/// the origin (the polytope always contains the origin during EPA).
fn make_face(verts: &[SupportVertex], i: usize, j: usize, k: usize) -> Option<Face> {
    let vi = verts[i].v;
    let vj = verts[j].v;
    let vk = verts[k].v;
    let mut normal = (vj - vi).cross(vk - vi);
    let len_sq = normal.length_squared();
    if len_sq <= DUPLICATE_EPSILON_SQ {
        return None;
    }
    normal /= len_sq.sqrt();
    let mut indices = [i, j, k];
    let mut distance = normal.dot(vi);
    if distance < 0.0 {
        // Flip winding so the normal points away from the origin.
        normal = -normal;
        distance = -distance;
        indices.swap(1, 2);
    }
    Some(Face {
        indices,
        normal,
        distance,
        dead: false,
    })
}

/// Expands the initial tetrahedron toward the origin and reads off the closest
/// boundary face as the minimum-translation contact.
fn epa<A: SupportMap, B: SupportMap>(
    a: &A,
    b: &B,
    simplex: [SupportVertex; 4],
) -> Option<Contact> {
    let mut verts: Vec<SupportVertex> = simplex.to_vec();
    let mut faces: Vec<Face> = Vec::with_capacity(8);
    for &(i, j, k) in &[(0usize, 1usize, 2usize), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
        faces.push(make_face(&verts, i, j, k)?);
    }

    for _ in 0..EPA_MAX_ITERATIONS {
        // Every face is a dead end: fall back to the closest non-degenerate face.
        let Some(closest) = closest_live_face(&faces) else {
            break;
        };
        let face = faces[closest];
        let s = support(a, b, face.normal);
        let projected = s.v.dot(face.normal);

        if projected - face.distance < EPA_TOLERANCE {
            return Some(build_contact(&verts, &face));
        }
        if verts
            .iter()
            .any(|q| (q.v - s.v).length_squared() < DUPLICATE_EPSILON_SQ)
        {
            // The outward support repeats an existing vertex, so this face
            // cannot grow. Retire it and let EPA search the remaining faces.
            faces[closest].dead = true;
            continue;
        }

        // Remove every face the new vertex can see and collect the horizon.
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut kept: Vec<Face> = Vec::with_capacity(faces.len() + 2);
        for f in &faces {
            let visible = f.normal.dot(s.v - verts[f.indices[0]].v) > 0.0;
            if visible {
                add_horizon_edge(&mut horizon, f.indices[0], f.indices[1]);
                add_horizon_edge(&mut horizon, f.indices[1], f.indices[2]);
                add_horizon_edge(&mut horizon, f.indices[2], f.indices[0]);
            } else {
                kept.push(*f);
            }
        }

        let new_index = verts.len();
        verts.push(s);
        for (p, q) in horizon {
            if let Some(face) = make_face(&verts, p, q, new_index) {
                kept.push(face);
            }
        }
        faces = kept;
    }

    // Iteration budget exhausted: return the closest non-degenerate face as
    // the best available approximation of the minimum-translation vector.
    let best = closest_nondegenerate_face(&faces).or_else(|| closest_face(&faces))?;
    Some(build_contact(&verts, &faces[best]))
}

/// Adds a directed edge to the horizon, cancelling it against an existing
/// opposite edge so only boundary edges of the visible region survive.
fn add_horizon_edge(horizon: &mut Vec<(usize, usize)>, p: usize, q: usize) {
    if let Some(pos) = horizon.iter().position(|&(a, b)| a == q && b == p) {
        horizon.swap_remove(pos);
    } else {
        horizon.push((p, q));
    }
}

/// Returns the index of the closest face that is still expandable.
fn closest_live_face(faces: &[Face]) -> Option<usize> {
    let mut best = None;
    let mut best_dist = f32::INFINITY;
    for (idx, f) in faces.iter().enumerate() {
        if !f.dead && f.distance < best_dist {
            best_dist = f.distance;
            best = Some(idx);
        }
    }
    best
}

/// Returns the closest face whose plane sits a non-degenerate distance from
/// the origin.
fn closest_nondegenerate_face(faces: &[Face]) -> Option<usize> {
    let mut best = None;
    let mut best_dist = f32::INFINITY;
    for (idx, f) in faces.iter().enumerate() {
        if f.distance > DEGENERATE_DISTANCE && f.distance < best_dist {
            best_dist = f.distance;
            best = Some(idx);
        }
    }
    best
}

/// Returns the index of the face closest to the origin.
fn closest_face(faces: &[Face]) -> Option<usize> {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (idx, f) in faces.iter().enumerate() {
        if f.distance < best_dist {
            best_dist = f.distance;
            best = idx;
        }
    }
    if best_dist.is_finite() {
        Some(best)
    } else {
        None
    }
}

/// Projects the origin onto the closest face and interpolates the per-shape
/// witness points from the face's barycentric coordinates.
fn build_contact(verts: &[SupportVertex], face: &Face) -> Contact {
    let [i, j, k] = face.indices;
    let projection = face.normal * face.distance;
    let (u, v, w) = barycentric(verts[i].v, verts[j].v, verts[k].v, projection);
    let point_a = verts[i].a * u + verts[j].a * v + verts[k].a * w;
    let point_b = verts[i].b * u + verts[j].b * v + verts[k].b * w;
    Contact {
        normal: face.normal,
        depth: face.distance,
        point_a,
        point_b,
    }
}

/// Barycentric coordinates of `p` with respect to triangle `(a, b, c)`.
fn barycentric(a: Vec3, b: Vec3, c: Vec3, p: Vec3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= DUPLICATE_EPSILON_SQ {
        return (1.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    (1.0 - v - w, v, w)
}

#[cfg(test)]
mod tests {
    use super::{gjk_contact, Contact};
    use crate::bounding::{Aabb, BoundingSphere, Obb};
    use glam::{Quat, Vec3};

    #[test]
    fn separated_spheres_report_no_contact() {
        let a = BoundingSphere::new(Vec3::ZERO, 1.0);
        let b = BoundingSphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(gjk_contact(&a, &b).is_none());
    }

    #[test]
    fn overlapping_boxes_recover_depth_normal_and_witnesses() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        // Overlap of 0.5 along +x.
        let b = Aabb::new(Vec3::new(0.5, -1.0, -1.0), Vec3::new(2.5, 1.0, 1.0));
        let Contact {
            normal,
            depth,
            point_a,
            point_b,
        } = gjk_contact(&a, &b).expect("boxes overlap");
        assert!((depth - 0.5).abs() < 1.0e-2, "depth = {depth}");
        assert!(normal.dot(Vec3::X).abs() > 0.99, "normal = {normal:?}");
        // Witnesses lie on the touching faces: A's +x face at x = 1, B's -x
        // face at x = 0.5. The normal convention makes `point_a` the surface
        // point on `a`, so it sits on the shared overlap region.
        assert!((point_a.x - 1.0).abs() < 5.0e-2, "point_a = {point_a:?}");
        assert!((point_b.x - 0.5).abs() < 5.0e-2, "point_b = {point_b:?}");
    }

    #[test]
    fn separating_vector_pushes_boxes_apart() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let b = Aabb::new(Vec3::new(0.5, -1.0, -1.0), Vec3::new(2.5, 1.0, 1.0));
        let contact = gjk_contact(&a, &b).expect("overlap");
        // Translate `a` by -normal*depth and confirm the shapes separate: the
        // moved A face (originally x = 1) lands on B's near face at x = 0.5.
        let moved_face_x = 1.0 + (-contact.normal * contact.depth).x;
        assert!((moved_face_x - 0.5).abs() < 1.0e-2, "moved = {moved_face_x}");
    }

    #[test]
    fn overlapping_boxes_recover_axis_depth() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        // Shifted +x by 1.5 so the two unit boxes overlap by 0.5 along x.
        let b = Aabb::new(Vec3::new(0.5, -1.0, -1.0), Vec3::new(2.5, 1.0, 1.0));
        let contact = gjk_contact(&a, &b).expect("boxes overlap");
        assert!((contact.depth - 0.5).abs() < 1.0e-2, "depth = {}", contact.depth);
        assert!(contact.normal.dot(Vec3::X).abs() > 0.99, "normal = {:?}", contact.normal);
    }

    #[test]
    fn rotated_obb_contact_is_shallow() {
        let a = Obb::new(Vec3::ZERO, Vec3::splat(0.5), Quat::IDENTITY);
        let b = Obb::new(
            Vec3::new(0.9, 0.0, 0.0),
            Vec3::splat(0.5),
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_4),
        );
        let contact = gjk_contact(&a, &b).expect("rotated boxes overlap");
        assert!(contact.depth > 0.0, "depth = {}", contact.depth);
        assert!(contact.depth < 0.5, "depth = {}", contact.depth);
        assert!(contact.normal.is_normalized(), "normal = {:?}", contact.normal);
    }
}

