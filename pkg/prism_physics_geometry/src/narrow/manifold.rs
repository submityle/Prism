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

use crate::bounding::{Aabb, Capsule, Obb};
use crate::narrow::closest_point::{closest_point_on_segment, closest_points_segment_segment};
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

/// When neither candidate face is at least this aligned with the contact
/// normal, the contact is an edge-edge (or edge-vertex) touch rather than a
/// face-face one, so face clipping cannot localise it well and we fall back to
/// an analytic closest-edge solve. `cos(~18 deg)` keeps genuine face contacts
/// (including lightly tilted resting boxes) on the clipping path.
const FACE_ALIGN_THRESHOLD: f32 = 0.95;

/// When the absolute cosine between a capsule's core segment and the box
/// contact normal exceeds this, the capsule meets the face end-on (nearly
/// perpendicular to it), so a single deepest point is the honest manifold.
/// Below it the segment lies along the face and a two-point manifold is sought.
const CAPSULE_END_ON_THRESHOLD: f32 = 0.9;

/// Absolute cosine between two capsule core directions above which they are
/// treated as parallel, enabling a two-point manifold spanning their overlap.
/// `cos(~5.7 deg)` keeps genuinely skew pairs on the single closest-pair path.
const CAPSULE_PARALLEL_THRESHOLD: f32 = 0.995;

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

    // Edge-edge / edge-vertex contact: neither face lies flat against the
    // contact plane, so clipping would place the point at a face corner rather
    // than the true crossing. Solve the closest edge pair analytically instead.
    if align_a.max(align_b) < FACE_ALIGN_THRESHOLD
        && let Some(m) = edge_edge_contact(&face_a, &face_b, normal, contact.depth)
    {
        return Some(m);
    }

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

/// Localises an edge-edge (or edge-vertex) contact by finding the closest pair
/// of face edges between the two candidate faces.
///
/// Face clipping collapses such contacts onto a face corner, biasing both the
/// position and the recovered depth. Here we test every edge of `face_a`
/// against every edge of `face_b` with [`closest_points_segment_segment`] and
/// keep the pair with the smallest gap; ties (common when edges truly cross and
/// the gap is ~0) are broken toward the most interior parameters so the point
/// lands at the crossing rather than at a shared endpoint. The penetration
/// depth is taken from the EPA witness, which is accurate for the normal even
/// when the witness *position* is not.
fn edge_edge_contact(
    face_a: &FacePolygon,
    face_b: &FacePolygon,
    normal: Vec3,
    depth: f32,
) -> Option<ContactManifold> {
    let va = &face_a.vertices;
    let vb = &face_b.vertices;
    if va.len() < 2 || vb.len() < 2 {
        return None;
    }

    let mut best_pos = Vec3::ZERO;
    let mut best_dist2 = f32::INFINITY;
    // Interiority score of the chosen pair: larger means the closest points sit
    // farther from both segment endpoints (used only to break near-ties).
    let mut best_interior = f32::NEG_INFINITY;
    let mut found = false;

    let na = va.len();
    let nb = vb.len();
    for i in 0..na {
        let a0 = va[i];
        let a1 = va[(i + 1) % na];
        if (a1 - a0).length_squared() < EDGE_EPSILON_SQ {
            continue;
        }
        for j in 0..nb {
            let b0 = vb[j];
            let b1 = vb[(j + 1) % nb];
            if (b1 - b0).length_squared() < EDGE_EPSILON_SQ {
                continue;
            }
            let cp = closest_points_segment_segment(a0, a1, b0, b1);
            // How far the closest params sit from either endpoint; higher is a
            // more credible interior crossing.
            let interior = cp.s.min(1.0 - cp.s).min(cp.t.min(1.0 - cp.t));
            // Accept a strictly closer pair, or an equally close but more
            // interior one (crossing edges all report ~0 distance).
            let closer = cp.distance_squared < best_dist2 - EDGE_EPSILON_SQ;
            let tie = (cp.distance_squared - best_dist2).abs() <= EDGE_EPSILON_SQ;
            if closer || (tie && interior > best_interior) {
                best_dist2 = cp.distance_squared.min(best_dist2);
                best_interior = interior;
                best_pos = (cp.c1 + cp.c2) * 0.5;
                found = true;
            }
        }
    }

    if !found {
        return None;
    }

    Some(ContactManifold {
        normal,
        points: alloc::vec![ManifoldPoint {
            position: best_pos,
            depth: depth.max(0.0),
        }],
    })
}

/// Builds a contact manifold between a capsule (shape `a`) and an oriented box
/// (shape `b`), producing a stable two-point manifold when the capsule's core
/// segment lies roughly parallel to the box's contact face.
///
/// A single deepest contact ([`Obb::capsule_contact`]) is enough to
/// depenetrate, but a capsule resting lengthwise on a surface pivots about one
/// point and see-saws unless both ends are constrained. This clips the core
/// segment to the prism above the box's contact face and resolves each clipped
/// end as a sphere, keeping only ends that actually penetrate. End-on
/// (near-perpendicular) and vertex contacts fall back to the single deepest
/// point. The manifold `normal` points from the capsule toward the box.
pub fn capsule_box_manifold(capsule: &Capsule, obb: &Obb) -> Option<ContactManifold> {
    let (deep_point, n_box, deep_depth) =
        obb.capsule_contact(capsule.a, capsule.b, capsule.radius)?;
    // `Obb::capsule_contact` reports the direction the capsule must move to
    // separate (box -> capsule); the manifold normal is capsule -> box.
    let normal = -n_box;

    let seg = capsule.b - capsule.a;
    let seg_len2 = seg.length_squared();
    // Sphere-like capsule, or an end-on contact: a single point is honest.
    if seg_len2 < EDGE_EPSILON_SQ {
        return one_point_manifold(deep_point, normal, deep_depth);
    }
    let seg_dir = seg * seg_len2.sqrt().recip();
    if seg_dir.dot(n_box).abs() > CAPSULE_END_ON_THRESHOLD {
        return one_point_manifold(deep_point, normal, deep_depth);
    }

    // Clip the core segment to the infinite prism over the box's contact face.
    let face = obb.best_face(n_box);
    if face.vertices.len() < 3 {
        return one_point_manifold(deep_point, normal, deep_depth);
    }
    let mut t_lo = 0.0f32;
    let mut t_hi = 1.0f32;
    let fv = &face.vertices;
    let n = fv.len();
    for i in 0..n {
        let e0 = fv[i];
        let e1 = fv[(i + 1) % n];
        let edge = e1 - e0;
        if edge.length_squared() < EDGE_EPSILON_SQ {
            continue;
        }
        let side = edge.cross(face.normal).normalize_or_zero();
        if side.length_squared() < 0.5 {
            continue;
        }
        // Keep the portion with side.dot(point) <= off (inside the face span).
        let off = side.dot(e0);
        let num = off - side.dot(capsule.a);
        let den = side.dot(seg);
        if den > f32::EPSILON {
            t_hi = t_hi.min(num / den);
        } else if den < -f32::EPSILON {
            t_lo = t_lo.max(num / den);
        } else if num < 0.0 {
            // Segment runs parallel to this side plane and entirely outside it.
            return one_point_manifold(deep_point, normal, deep_depth);
        }
    }

    if t_lo > t_hi {
        return one_point_manifold(deep_point, normal, deep_depth);
    }

    // Resolve each clipped end as a sphere; keep ends that truly penetrate.
    let mut points = Vec::with_capacity(2);
    for &t in &[t_lo, t_hi] {
        let sample = capsule.a + seg * t;
        if let Some((cp, _, depth)) = obb.sphere_contact(sample, capsule.radius) {
            points.push(ManifoldPoint {
                position: cp,
                depth: depth.max(0.0),
            });
        }
        // A zero-length clip span contributes only one sample.
        if (t_hi - t_lo).abs() <= CONTACT_SLOP {
            break;
        }
    }

    if points.is_empty() {
        return one_point_manifold(deep_point, normal, deep_depth);
    }

    Some(ContactManifold { normal, points })
}

/// Builds a contact manifold between two capsules, producing a stable two-point
/// manifold when their core segments run parallel and their overlap spans a
/// length (as for stacked limbs or capsule chains in a ragdoll).
///
/// A single deepest contact ([`Capsule::contact`]) depenetrates the pair, but
/// two parallel capsules touch along a line segment; constraining only the
/// midpoint lets them scissor. Here, when the cores are near-parallel, the
/// overlapping interval along the shared axis is found and both ends are
/// resolved, keeping only ends that penetrate. Skew or sphere-like pairs fall
/// back to the single closest-pair contact. The normal points from `a` to `b`.
pub fn capsule_capsule_manifold(a: &Capsule, b: &Capsule) -> Option<ContactManifold> {
    let (deep_point, normal, deep_depth) = a.contact(b)?;

    let da = a.b - a.a;
    let db = b.b - b.a;
    let la2 = da.length_squared();
    let lb2 = db.length_squared();
    if la2 < EDGE_EPSILON_SQ || lb2 < EDGE_EPSILON_SQ {
        return one_point_manifold(deep_point, normal, deep_depth);
    }
    let la = la2.sqrt();
    let dir_a = da * la.recip();
    let dir_b = db * lb2.sqrt().recip();
    if dir_a.dot(dir_b).abs() < CAPSULE_PARALLEL_THRESHOLD {
        return one_point_manifold(deep_point, normal, deep_depth);
    }

    // Overlap interval of the two cores projected onto `a`'s axis (origin a.a).
    let so0 = (b.a - a.a).dot(dir_a);
    let so1 = (b.b - a.a).dot(dir_a);
    let lo = 0.0f32.max(so0.min(so1));
    let hi = la.min(so0.max(so1));
    if hi - lo <= CONTACT_SLOP {
        return one_point_manifold(deep_point, normal, deep_depth);
    }

    let sum = a.radius + b.radius;
    let mut points = Vec::with_capacity(2);
    for &t in &[lo, hi] {
        let pa = a.a + dir_a * t;
        let pb = closest_point_on_segment(pa, b.a, b.b);
        let gap = pb - pa;
        let dist = gap.length();
        let depth = sum - dist;
        if depth > 0.0 {
            let n = if dist > 1.0e-6 { gap * dist.recip() } else { normal };
            let surf_a = pa + n * a.radius;
            let surf_b = pb - n * b.radius;
            points.push(ManifoldPoint {
                position: (surf_a + surf_b) * 0.5,
                depth,
            });
        }
    }

    if points.len() < 2 {
        return one_point_manifold(deep_point, normal, deep_depth);
    }
    Some(ContactManifold { normal, points })
}

/// Wraps a single resolved contact point into a one-point manifold.
fn one_point_manifold(position: Vec3, normal: Vec3, depth: f32) -> Option<ContactManifold> {
    Some(ContactManifold {
        normal,
        points: alloc::vec![ManifoldPoint {
            position,
            depth: depth.max(0.0),
        }],
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
    use crate::bounding::{Aabb, Capsule, Obb};
    use super::{capsule_box_manifold, capsule_capsule_manifold};
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

    #[test]
    fn crossing_edges_localise_at_true_intersection() {
        // Box A tilted 45 deg about X: its top ridge runs along world X at
        // y ~= sqrt(2). Box B tilted 45 deg about Z and placed above: its
        // bottom ridge runs along world Z. The two ridges cross perpendicularly
        // above the origin, so the analytic contact point is x = 0, z = 0.
        let a = Obb::new(
            Vec3::ZERO,
            Vec3::splat(1.0),
            Quat::from_rotation_x(core::f32::consts::FRAC_PI_4),
        );
        let b = Obb::new(
            Vec3::new(0.0, 2.7, 0.0),
            Vec3::splat(1.0),
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_4),
        );
        let m = contact_manifold(&a, &b).expect("crossing ridges overlap");
        // Edge-edge contact localises to a single crossing point.
        assert_eq!(m.points.len(), 1, "edge-edge yields a single point");
        assert!(m.normal.y > 0.9, "normal is roughly +Y: {:?}", m.normal);
        let pos = m.points[0].position;
        // The old face-clipping fallback landed the point at z ~= 0.128 (a face
        // corner). The analytic solve must sit on the true crossing (x=0, z=0).
        assert!(pos.x.abs() < 2.0e-2, "x near crossing: {}", pos.x);
        assert!(pos.z.abs() < 2.0e-2, "z near crossing: {}", pos.z);
        assert!(m.points[0].depth > 0.0, "contact penetrates");
    }

    #[test]
    fn horizontal_capsule_resting_on_box_is_two_point() {
        // Unit box centred at origin (top face at y = 1). A capsule lying along
        // world X just above it, pressed down so its lower surface sinks 0.1
        // into the face.
        let obb = Obb::new(Vec3::ZERO, Vec3::splat(1.0), Quat::IDENTITY);
        let capsule = Capsule::new(
            Vec3::new(-0.5, 1.4, 0.0),
            Vec3::new(0.5, 1.4, 0.0),
            0.5,
        );
        let m = capsule_box_manifold(&capsule, &obb).expect("resting contact");
        // Normal points capsule -> box, i.e. downward.
        assert!(m.normal.y < -0.9, "normal points down: {:?}", m.normal);
        assert_eq!(m.points.len(), 2, "lengthwise rest yields two points");
        for p in &m.points {
            assert!(p.depth > 0.0, "each end penetrates: {}", p.depth);
            assert!((p.position.y - 1.0).abs() < 1e-4, "point on top face");
        }
        // The two points straddle the capsule span along X.
        let x0 = m.points[0].position.x;
        let x1 = m.points[1].position.x;
        assert!(
            x0.min(x1) < -0.3 && x0.max(x1) > 0.3,
            "points straddle span: {x0}, {x1}"
        );
    }

    #[test]
    fn vertical_capsule_on_box_is_single_point() {
        // Capsule standing end-on above the box: only the lower cap touches.
        let obb = Obb::new(Vec3::ZERO, Vec3::splat(1.0), Quat::IDENTITY);
        let capsule = Capsule::new(
            Vec3::new(0.0, 1.4, 0.0),
            Vec3::new(0.0, 3.4, 0.0),
            0.5,
        );
        let m = capsule_box_manifold(&capsule, &obb).expect("end-on contact");
        assert_eq!(m.points.len(), 1, "end-on contact is a single point");
        assert!(m.normal.y < -0.9);
        assert!(m.points[0].depth > 0.0);
    }

    #[test]
    fn capsule_overhanging_box_edge_keeps_only_supported_end() {
        // Capsule lies along X but shifted so one end hangs past the box's +X
        // edge. The clipped span should drop the unsupported end, yielding a
        // single contact where the capsule still rests on the face.
        let obb = Obb::new(Vec3::ZERO, Vec3::splat(1.0), Quat::IDENTITY);
        let capsule = Capsule::new(
            Vec3::new(0.5, 1.4, 0.0),
            Vec3::new(3.0, 1.4, 0.0),
            0.5,
        );
        let m = capsule_box_manifold(&capsule, &obb).expect("overhang contact");
        assert!(m.normal.y < -0.9);
        for p in &m.points {
            assert!(p.position.x <= 1.0 + 1e-3, "contact stays over the face");
            assert!(p.depth > 0.0);
        }
    }

    #[test]
    fn parallel_capsules_yield_two_point_manifold() {
        // Two X-aligned capsules stacked in Y, overlapping by 0.2.
        let a = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5);
        let b = Capsule::new(Vec3::new(-1.0, 0.8, 0.0), Vec3::new(1.0, 0.8, 0.0), 0.5);
        let m = capsule_capsule_manifold(&a, &b).expect("overlap");
        assert!(m.normal.y > 0.9, "normal a->b points up: {:?}", m.normal);
        assert_eq!(m.points.len(), 2, "parallel cores yield two points");
        for p in &m.points {
            assert!((p.depth - 0.2).abs() < 1e-3, "uniform depth: {}", p.depth);
        }
        let x0 = m.points[0].position.x;
        let x1 = m.points[1].position.x;
        assert!(x0.min(x1) < -0.5 && x0.max(x1) > 0.5, "span: {x0}, {x1}");
    }

    #[test]
    fn perpendicular_capsules_are_single_point() {
        // X-aligned under a Z-aligned capsule crossing above the origin.
        let a = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5);
        let b = Capsule::new(Vec3::new(0.0, 0.8, -1.0), Vec3::new(0.0, 0.8, 1.0), 0.5);
        let m = capsule_capsule_manifold(&a, &b).expect("crossing overlap");
        assert_eq!(m.points.len(), 1, "skew cores give a single point");
        assert!(m.points[0].depth > 0.0);
    }

    #[test]
    fn partially_overlapping_parallel_capsules_clip_to_overlap() {
        // Offset so the shared span is only x in [0, 1].
        let a = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5);
        let b = Capsule::new(Vec3::new(0.0, 0.8, 0.0), Vec3::new(2.0, 0.8, 0.0), 0.5);
        let m = capsule_capsule_manifold(&a, &b).expect("overlap");
        assert_eq!(m.points.len(), 2);
        for p in &m.points {
            assert!(
                p.position.x >= -1e-3 && p.position.x <= 1.0 + 1e-3,
                "contact confined to shared span: {}",
                p.position.x
            );
            assert!(p.depth > 0.0);
        }
    }
}
