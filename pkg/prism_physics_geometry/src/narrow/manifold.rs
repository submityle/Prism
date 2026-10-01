//! Multi-point contact manifold generation for overlapping convex polytopes.
//!
//! A single penetration witness (see [`crate::narrow::gjk_contact`]) is enough
//! to push two shapes apart, but a stable rigid-body solver needs several
//! simultaneous contact points so a resting box does not jitter or tip. This
//! module takes the separating normal from EPA and recovers a full
//! face-face manifold by clipping the *incident* face of one polytope against
//! the side planes of the *reference* face of the other (the classic
//! Sutherland-Hodgman clipping manifold).
//!
//! The algorithm is shape-agnostic over the [`ClipShape`] trait, which reports
//! the polygonal face whose outward normal is most aligned with a query
//! direction. Boxes ([`Aabb`]/[`Obb`]) implement it directly.
//!
//! This is a clean-room implementation of the publicly documented clipping
//! manifold technique and contains no Unreal Engine source or derived code.

use alloc::vec::Vec;
use glam::Vec3;

use crate::bounding::{Aabb, Obb};
use crate::narrow::epa::gjk_contact;
use crate::narrow::support::SupportMap;

/// Points on a clipped polygon closer to the reference plane than this (in
/// world units) are kept as contact points. A small positive slop admits
/// grazing contacts that the solver still wants to see.
const CONTACT_SLOP: f32 = 1.0e-3;

/// Squared length below which a candidate face edge is treated as degenerate.
const EDGE_EPSILON_SQ: f32 = 1.0e-12;

/// Maximum points retained in a manifold. Four points fully constrain a
/// face-face contact; extras from clipping are reduced away.
const MAX_MANIFOLD_POINTS: usize = 4;

/// A planar convex face of a polytope, wound counter-clockwise when viewed
/// from outside the shape (looking against `normal`).
#[derive(Clone, Debug)]
pub struct FacePolygon {
    /// Outward unit normal of the face.
    pub normal: Vec3,
    /// Face vertices in counter-clockwise order (as seen from outside).
    pub vertices: Vec<Vec3>,
}

/// A convex shape that can report the polygonal face best aligned with a
/// direction, enabling face-clipping contact manifolds.
pub trait ClipShape: SupportMap {
    /// Returns the face whose outward normal is most aligned with `dir`.
    fn best_face(&self, dir: Vec3) -> FacePolygon;
}

/// A single contact point of a manifold.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ManifoldPoint {
    /// Contact position in world space (on the incident surface).
    pub position: Vec3,
    /// Penetration depth along the manifold normal (non-negative).
    pub depth: f32,
}

/// A face-face contact manifold between two overlapping convex shapes.
#[derive(Clone, Debug)]
pub struct ContactManifold {
    /// Unit contact normal pointing from shape `a` toward shape `b`.
    ///
    /// Separating the pair requires translating `a` along `-normal`.
    pub normal: Vec3,
    /// One to [`MAX_MANIFOLD_POINTS`] contact points.
    pub points: Vec<ManifoldPoint>,
}

/// Builds the four counter-clockwise corners of the box face most aligned with
/// `dir`, given a box's `center`, orthonormal `axes` and `half_extents`.
fn box_best_face(center: Vec3, axes: [Vec3; 3], half: Vec3, dir: Vec3) -> FacePolygon {
    let he = [half.x, half.y, half.z];
    // Pick the (axis, sign) whose signed face normal is most aligned with dir.
    let mut best_axis = 0usize;
    let mut best_sign = 1.0f32;
    let mut best_align = f32::NEG_INFINITY;
    for (i, axis) in axes.iter().enumerate() {
        let proj = dir.dot(*axis);
        let sign = if proj >= 0.0 { 1.0 } else { -1.0 };
        let align = proj * sign;
        if align > best_align {
            best_align = align;
            best_axis = i;
            best_sign = sign;
        }
    }
    let i = best_axis;
    let j = (i + 1) % 3;
    let k = (i + 2) % 3;
    let normal = axes[i] * best_sign;
    let face_center = center + normal * he[i];
    let uj = axes[j] * he[j];
    let uk = axes[k] * he[k];
    // Four corners around the face.
    let mut vertices = alloc::vec![
        face_center + uj + uk,
        face_center - uj + uk,
        face_center - uj - uk,
        face_center + uj - uk,
    ];
    // Ensure counter-clockwise winding relative to the outward normal.
    let winding = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[1]);
    if winding.dot(normal) < 0.0 {
        vertices.reverse();
    }
    FacePolygon { normal, vertices }
}

impl ClipShape for Obb {
    fn best_face(&self, dir: Vec3) -> FacePolygon {
        box_best_face(self.center, self.axes(), self.half_extents, dir)
    }
}

impl ClipShape for Aabb {
    fn best_face(&self, dir: Vec3) -> FacePolygon {
        let axes = [Vec3::X, Vec3::Y, Vec3::Z];
        box_best_face(self.center(), axes, self.half_extents(), dir)
    }
}

/// Clips the convex `polygon` against a half-space, keeping the portion where
/// `normal.dot(p) <= offset`. Crossings are linearly interpolated.
fn clip_against_plane(polygon: &[Vec3], normal: Vec3, offset: f32) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(polygon.len() + 1);
    if polygon.is_empty() {
        return out;
    }
    let n = polygon.len();
    for i in 0..n {
        let current = polygon[i];
        let next = polygon[(i + 1) % n];
        let dist_c = normal.dot(current) - offset;
        let dist_n = normal.dot(next) - offset;
        let inside_c = dist_c <= 0.0;
        let inside_n = dist_n <= 0.0;
        if inside_c {
            out.push(current);
        }
        // Edge crosses the plane: insert the intersection point.
        if inside_c != inside_n {
            let denom = dist_c - dist_n;
            if denom.abs() > f32::EPSILON {
                let t = dist_c / denom;
                out.push(current + (next - current) * t);
            }
        }
    }
    out
}

/// Reduces a clipped point set to at most [`MAX_MANIFOLD_POINTS`] by keeping the
/// deepest point and the points that maximise the manifold's spread, which
/// preserves the stabilising moment arm for the solver.
fn reduce_points(mut points: Vec<ManifoldPoint>) -> Vec<ManifoldPoint> {
    if points.len() <= MAX_MANIFOLD_POINTS {
        return points;
    }

    let mut kept = Vec::with_capacity(MAX_MANIFOLD_POINTS);

    // 1. Deepest point anchors the manifold.
    let mut deepest = 0usize;
    for (idx, p) in points.iter().enumerate() {
        if p.depth > points[deepest].depth {
            deepest = idx;
        }
    }
    kept.push(points.swap_remove(deepest));

    // 2. Point farthest from the first anchor.
    let a = kept[0].position;
    let mut far = 0usize;
    let mut far_d2 = f32::NEG_INFINITY;
    for (idx, p) in points.iter().enumerate() {
        let d2 = (p.position - a).length_squared();
        if d2 > far_d2 {
            far_d2 = d2;
            far = idx;
        }
    }
    kept.push(points.swap_remove(far));

    // 3. Point maximising triangle area with the current edge (most positive
    //    signed area along the normal).
    let b = kept[1].position;
    let edge = b - a;
    let mut best_pos = 0usize;
    let mut best_pos_area = 0.0f32;
    let mut best_neg = 0usize;
    let mut best_neg_area = 0.0f32;
    for (idx, p) in points.iter().enumerate() {
        let area = edge.cross(p.position - a).length();
        let side = edge.cross(p.position - a);
        // Use an arbitrary reference axis to split into two sides.
        let signed = side.dot(Vec3::Y) + side.dot(Vec3::X) + side.dot(Vec3::Z);
        if signed >= 0.0 {
            if area > best_pos_area {
                best_pos_area = area;
                best_pos = idx;
            }
        } else if area > best_neg_area {
            best_neg_area = area;
            best_neg = idx;
        }
    }

    // Collect the two spread points (guard against picking the same index).
    let mut extra = Vec::new();
    if best_pos_area > 0.0 {
        extra.push(best_pos);
    }
    if best_neg_area > 0.0 && best_neg != best_pos {
        extra.push(best_neg);
    }
    extra.sort_unstable();
    for idx in extra.into_iter().rev() {
        if kept.len() < MAX_MANIFOLD_POINTS {
            kept.push(points.swap_remove(idx));
        }
    }

    // Backfill with deepest remaining if still short.
    while kept.len() < MAX_MANIFOLD_POINTS && !points.is_empty() {
        let mut d = 0usize;
        for (idx, p) in points.iter().enumerate() {
            if p.depth > points[d].depth {
                d = idx;
            }
        }
        kept.push(points.swap_remove(d));
    }

    kept
}

/// Builds a face-clipping contact manifold between two overlapping convex
/// polytopes `a` and `b`.
///
/// Returns `Some(manifold)` with one to four contact points when the shapes
/// penetrate, or `None` when they are separate or only grazing. The manifold
/// `normal` points from `a` toward `b`, matching [`crate::narrow::Contact`].
pub fn contact_manifold<A: ClipShape, B: ClipShape>(a: &A, b: &B) -> Option<ContactManifold> {
    let contact = gjk_contact(a, b)?;
    let normal = contact.normal;

    // Reference/incident face selection: the face whose outward normal is most
    // parallel to the contact axis becomes the reference; we clip the incident
    // face against its side planes.
    let face_a = a.best_face(normal);
    let face_b = b.best_face(-normal);

    let align_a = face_a.normal.dot(normal);
    let align_b = face_b.normal.dot(-normal);

    let (reference, incident) = if align_a >= align_b {
        (&face_a, &face_b)
    } else {
        (&face_b, &face_a)
    };

    if reference.vertices.len() < 3 || incident.vertices.len() < 2 {
        return single_point_fallback(contact.point_a, contact.point_b, normal, contact.depth);
    }

    // Clip the incident polygon against each side plane of the reference face.
    // A side plane is perpendicular to the reference face and passes through a
    // reference edge; its inward direction keeps points over the face.
    let rn = reference.normal;
    let mut clipped = incident.vertices.clone();
    let rv = &reference.vertices;
    let count = rv.len();
    for i in 0..count {
        let a0 = rv[i];
        let a1 = rv[(i + 1) % count];
        let edge = a1 - a0;
        if edge.length_squared() < EDGE_EPSILON_SQ {
            continue;
        }
        // Outward side-plane normal (points away from the face interior).
        let side_normal = edge.cross(rn).normalize_or_zero();
        if side_normal.length_squared() < 0.5 {
            continue;
        }
        let offset = side_normal.dot(a0);
        clipped = clip_against_plane(&clipped, side_normal, offset);
        if clipped.is_empty() {
            break;
        }
    }

    if clipped.is_empty() {
        return single_point_fallback(contact.point_a, contact.point_b, normal, contact.depth);
    }

    // Keep clipped points that sit at or behind the reference face plane.
    let ref_offset = rn.dot(reference.vertices[0]);
    let mut points = Vec::with_capacity(clipped.len());
    for p in &clipped {
        let separation = rn.dot(*p) - ref_offset;
        if separation <= CONTACT_SLOP {
            points.push(ManifoldPoint {
                position: *p,
                depth: (-separation).max(0.0),
            });
        }
    }

    if points.is_empty() {
        return single_point_fallback(contact.point_a, contact.point_b, normal, contact.depth);
    }

    Some(ContactManifold {
        normal,
        points: reduce_points(points),
    })
}

/// Produces a one-point manifold from the EPA witness pair when face clipping
/// cannot form a polygon (e.g. edge or vertex contact).
fn single_point_fallback(
    point_a: Vec3,
    point_b: Vec3,
    normal: Vec3,
    depth: f32,
) -> Option<ContactManifold> {
    let position = (point_a + point_b) * 0.5;
    Some(ContactManifold {
        normal,
        points: alloc::vec![ManifoldPoint {
            position,
            depth: depth.max(0.0),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::{contact_manifold, ClipShape};
    use crate::bounding::{Aabb, Obb};
    use glam::{Quat, Vec3};

    #[test]
    fn aabb_best_face_points_along_query() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let face = b.best_face(Vec3::Y);
        assert!(face.normal.abs_diff_eq(Vec3::Y, 1e-6));
        assert_eq!(face.vertices.len(), 4);
        // All face vertices sit on the +Y plane.
        for v in &face.vertices {
            assert!((v.y - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn stacked_aabbs_produce_four_point_manifold() {
        // Lower box fixed; upper box overlapping it slightly from above.
        let lower = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 0.0, 1.0));
        let upper = Aabb::new(Vec3::new(-1.0, -0.1, -1.0), Vec3::new(1.0, 0.9, 1.0));
        let m = contact_manifold(&lower, &upper).expect("boxes overlap");
        // Normal from lower toward upper is +Y.
        assert!(m.normal.y.abs() > 0.9, "normal mostly vertical: {:?}", m.normal);
        assert_eq!(m.points.len(), 4, "flat face-face contact yields 4 points");
        for p in &m.points {
            assert!(p.depth > 0.0, "each point penetrates");
            assert!((p.depth - 0.1).abs() < 1e-2, "depth ~0.1, got {}", p.depth);
        }
    }

    #[test]
    fn separated_boxes_have_no_manifold() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let b = Aabb::new(Vec3::splat(2.0), Vec3::splat(4.0));
        assert!(contact_manifold(&a, &b).is_none());
    }

    #[test]
    fn obb_face_contact_is_multipoint() {
        // Axis-aligned OBB resting slightly inside another along +Y.
        let lower = Obb::new(Vec3::ZERO, Vec3::splat(1.0), Quat::IDENTITY);
        let upper = Obb::new(Vec3::new(0.0, 1.9, 0.0), Vec3::splat(1.0), Quat::IDENTITY);
        let m = contact_manifold(&lower, &upper).expect("obb overlap");
        assert!(m.normal.y.abs() > 0.9);
        assert!(m.points.len() >= 2, "face-face yields multiple points");
        for p in &m.points {
            assert!(p.depth > 0.0);
        }
    }

    #[test]
    fn rotated_obb_edge_contact_still_reports() {
        // Upper box rotated 45deg about Z so a wedge presses into the lower box.
        let lower = Obb::new(Vec3::ZERO, Vec3::splat(1.0), Quat::IDENTITY);
        let upper = Obb::new(
            Vec3::new(0.0, 1.8, 0.0),
            Vec3::splat(1.0),
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_4),
        );
        let m = contact_manifold(&lower, &upper).expect("rotated overlap");
        assert!(!m.points.is_empty());
        for p in &m.points {
            assert!(p.depth >= 0.0);
        }
    }
}
