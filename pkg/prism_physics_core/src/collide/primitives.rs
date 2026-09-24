//! Closed-form and polytope narrow-phase routines for the analytic primitives.
//!
//! This module implements [`generate_contact`], the narrow-phase dispatch that
//! turns an ordered pair of posed
//! [`ColliderShape`](crate::collider::ColliderShape)s into an optional
//! [`ContactManifold`]. Every supported shape pair is genuinely computed: there
//! are no stubs, and each pair either produces a manifold following the frozen
//! [`contact`](super::contact) conventions (unit normal from `a` toward `b`,
//! non-negative penetration, and the witness identity
//! `point_a == point_b + penetration * normal`) or returns `None` when the
//! shapes are separated beyond a small tolerance.
//!
//! # Supported pairs
//!
//! sphere-sphere, sphere-cuboid, sphere-capsule, sphere-plane,
//! capsule-capsule, capsule-cuboid, capsule-plane, cuboid-plane, and
//! cuboid-cuboid. Swapped argument orders are handled by the dispatch, which
//! flips the normal and swaps witness points so the `a -> b` convention always
//! holds for the actual argument order. Plane-plane returns `None`.
//!
//! # Provenance
//!
//! The algorithms used here are standard, publicly documented
//! computational-geometry techniques: closed-form sphere/capsule/plane distance
//! tests, closest points between line segments (Ericson, *Real-Time Collision
//! Detection*), the Separating-Axis Test over the 15 candidate axes of two
//! oriented boxes, and Sutherland-Hodgman polygon clipping for the box-box
//! face-contact patch. This code contains **no Unreal Engine source or derived
//! code**.

use core::cmp::Ordering;

use glam::Vec3;

use super::contact::{ContactManifold, ContactPoint, MAX_MANIFOLD_POINTS};
use crate::collider::ColliderShape;
use crate::math::transform::Isometry;
use crate::state::handle::BodyHandle;

/// Distance within which touching shapes are still reported as in contact.
const CONTACT_TOLERANCE: f32 = 1.0e-4;

/// Threshold below which a length is treated as numerically zero.
const GEOMETRIC_EPS: f32 = 1.0e-6;

/// Generates the contact manifold for a posed pair of collider shapes.
///
/// Returns `Some(manifold)` when the shapes overlap or touch within
/// [`CONTACT_TOLERANCE`], and `None` otherwise. The returned manifold follows
/// the frozen conventions documented on [`contact`](super::contact): the normal
/// is unit length and points from `a` toward `b`, penetration is non-negative,
/// and body handles are left as
/// [`BodyHandle::INVALID`](crate::state::handle::BodyHandle::INVALID) for the
/// caller to stamp.
#[must_use]
pub fn generate_contact(
    shape_a: &ColliderShape,
    pose_a: &Isometry,
    shape_b: &ColliderShape,
    pose_b: &Isometry,
) -> Option<ContactManifold> {
    match (shape_a, shape_b) {
        (ColliderShape::Sphere { radius: ra }, ColliderShape::Sphere { radius: rb }) => {
            sphere_sphere(*ra, pose_a, *rb, pose_b)
        }
        (ColliderShape::Sphere { radius }, ColliderShape::Cuboid { half_extents }) => {
            sphere_cuboid(*radius, pose_a, *half_extents, pose_b)
        }
        (ColliderShape::Cuboid { half_extents }, ColliderShape::Sphere { radius }) => {
            sphere_cuboid(*radius, pose_b, *half_extents, pose_a).map(flipped)
        }
        (
            ColliderShape::Sphere { radius },
            ColliderShape::Capsule {
                half_height,
                radius: rc,
            },
        ) => sphere_capsule(*radius, pose_a, *half_height, *rc, pose_b),
        (
            ColliderShape::Capsule {
                half_height,
                radius: rc,
            },
            ColliderShape::Sphere { radius },
        ) => sphere_capsule(*radius, pose_b, *half_height, *rc, pose_a).map(flipped),
        (ColliderShape::Sphere { radius }, ColliderShape::Plane { normal, offset }) => {
            sphere_plane(*radius, pose_a, *normal, *offset, pose_b)
        }
        (ColliderShape::Plane { normal, offset }, ColliderShape::Sphere { radius }) => {
            sphere_plane(*radius, pose_b, *normal, *offset, pose_a).map(flipped)
        }
        (
            ColliderShape::Capsule {
                half_height: ha,
                radius: rah,
            },
            ColliderShape::Capsule {
                half_height: hb,
                radius: rbh,
            },
        ) => capsule_capsule(*ha, *rah, pose_a, *hb, *rbh, pose_b),
        (
            ColliderShape::Capsule {
                half_height,
                radius,
            },
            ColliderShape::Cuboid { half_extents },
        ) => capsule_cuboid(*half_height, *radius, pose_a, *half_extents, pose_b),
        (
            ColliderShape::Cuboid { half_extents },
            ColliderShape::Capsule {
                half_height,
                radius,
            },
        ) => capsule_cuboid(*half_height, *radius, pose_b, *half_extents, pose_a).map(flipped),
        (
            ColliderShape::Capsule {
                half_height,
                radius,
            },
            ColliderShape::Plane { normal, offset },
        ) => capsule_plane(*half_height, *radius, pose_a, *normal, *offset, pose_b),
        (
            ColliderShape::Plane { normal, offset },
            ColliderShape::Capsule {
                half_height,
                radius,
            },
        ) => capsule_plane(*half_height, *radius, pose_b, *normal, *offset, pose_a).map(flipped),
        (ColliderShape::Cuboid { half_extents }, ColliderShape::Plane { normal, offset }) => {
            cuboid_plane(*half_extents, pose_a, *normal, *offset, pose_b)
        }
        (ColliderShape::Plane { normal, offset }, ColliderShape::Cuboid { half_extents }) => {
            cuboid_plane(*half_extents, pose_b, *normal, *offset, pose_a).map(flipped)
        }
        (
            ColliderShape::Cuboid { half_extents: hea },
            ColliderShape::Cuboid { half_extents: heb },
        ) => cuboid_cuboid(*hea, pose_a, *heb, pose_b),
        (ColliderShape::Plane { .. }, ColliderShape::Plane { .. }) => None,
    }
}

/// Reverses a manifold computed for the pair `(b, a)` so it describes `(a, b)`.
///
/// The normal is negated and each contact's witness points are swapped, which
/// preserves the `point_a == point_b + penetration * normal` identity.
fn flipped(manifold: ContactManifold) -> ContactManifold {
    let mut out = ContactManifold::new(manifold.body_a, manifold.body_b, -manifold.normal);
    for point in manifold.points() {
        out.push(ContactPoint::new(
            point.point_b,
            point.point_a,
            point.penetration,
        ));
    }
    out
}

/// Assembles a manifold from a normal and a list of witness/penetration tuples.
///
/// Returns `None` when the point list is empty.
fn manifold_from_points(normal: Vec3, points: &[(Vec3, Vec3, f32)]) -> Option<ContactManifold> {
    if points.is_empty() {
        return None;
    }
    let mut manifold = ContactManifold::new(BodyHandle::INVALID, BodyHandle::INVALID, normal);
    for &(point_a, point_b, penetration) in points {
        manifold.push(ContactPoint::new(point_a, point_b, penetration));
    }
    Some(manifold)
}

/// Picks an arbitrary unit vector, used when a contact normal is degenerate.
fn fallback_normal() -> Vec3 {
    Vec3::X
}

// ---------------------------------------------------------------------------
// Sphere pairs
// ---------------------------------------------------------------------------

/// Contact between two spheres, with the normal pointing from `a` toward `b`.
fn sphere_sphere(
    radius_a: f32,
    pose_a: &Isometry,
    radius_b: f32,
    pose_b: &Isometry,
) -> Option<ContactManifold> {
    let center_a = pose_a.translation;
    let center_b = pose_b.translation;
    let delta = center_b - center_a;
    let distance = delta.length();
    let sum = radius_a + radius_b;
    let penetration = sum - distance;
    if penetration < -CONTACT_TOLERANCE {
        return None;
    }
    let normal = if distance > GEOMETRIC_EPS {
        delta / distance
    } else {
        fallback_normal()
    };
    let point_a = center_a + normal * radius_a;
    let point_b = center_b - normal * radius_b;
    manifold_from_points(normal, &[(point_a, point_b, penetration)])
}

/// Contact between a sphere (`a`) and a cuboid (`b`).
fn sphere_cuboid(
    radius: f32,
    pose_sphere: &Isometry,
    half_extents: Vec3,
    pose_box: &Isometry,
) -> Option<ContactManifold> {
    let obb = Obb::new(half_extents, pose_box);
    let center = pose_sphere.translation;
    let (normal, point_a, point_b, penetration) = point_vs_box(center, radius, &obb)?;
    manifold_from_points(normal, &[(point_a, point_b, penetration)])
}

/// Contact between a sphere (`a`) and a capsule (`b`).
fn sphere_capsule(
    radius: f32,
    pose_sphere: &Isometry,
    half_height: f32,
    capsule_radius: f32,
    pose_capsule: &Isometry,
) -> Option<ContactManifold> {
    let center = pose_sphere.translation;
    let (seg0, seg1) = capsule_segment(half_height, pose_capsule);
    let on_axis = closest_point_on_segment(center, seg0, seg1);
    let delta = on_axis - center;
    let distance = delta.length();
    let sum = radius + capsule_radius;
    let penetration = sum - distance;
    if penetration < -CONTACT_TOLERANCE {
        return None;
    }
    let normal = if distance > GEOMETRIC_EPS {
        delta / distance
    } else {
        fallback_normal()
    };
    let point_a = center + normal * radius;
    let point_b = on_axis - normal * capsule_radius;
    manifold_from_points(normal, &[(point_a, point_b, penetration)])
}

/// Contact between a sphere (`a`) and a plane half-space (`b`).
fn sphere_plane(
    radius: f32,
    pose_sphere: &Isometry,
    plane_normal: Vec3,
    plane_offset: f32,
    pose_plane: &Isometry,
) -> Option<ContactManifold> {
    let (normal_world, offset_world) = plane_world(plane_normal, plane_offset, pose_plane);
    let center = pose_sphere.translation;
    let signed = normal_world.dot(center) - offset_world;
    let penetration = radius - signed;
    if penetration < -CONTACT_TOLERANCE {
        return None;
    }
    let normal = -normal_world;
    let point_a = center - normal_world * radius;
    let point_b = center - normal_world * signed;
    manifold_from_points(normal, &[(point_a, point_b, penetration)])
}

// ---------------------------------------------------------------------------
// Capsule pairs
// ---------------------------------------------------------------------------

/// Contact between two capsules via the closest points of their core segments.
fn capsule_capsule(
    half_height_a: f32,
    radius_a: f32,
    pose_a: &Isometry,
    half_height_b: f32,
    radius_b: f32,
    pose_b: &Isometry,
) -> Option<ContactManifold> {
    let (a0, a1) = capsule_segment(half_height_a, pose_a);
    let (b0, b1) = capsule_segment(half_height_b, pose_b);
    let (pa, pb) = closest_points_segments(a0, a1, b0, b1);
    let delta = pb - pa;
    let distance = delta.length();
    let sum = radius_a + radius_b;
    let penetration = sum - distance;
    if penetration < -CONTACT_TOLERANCE {
        return None;
    }
    let normal = if distance > GEOMETRIC_EPS {
        delta / distance
    } else {
        fallback_normal()
    };
    let point_a = pa + normal * radius_a;
    let point_b = pb - normal * radius_b;
    manifold_from_points(normal, &[(point_a, point_b, penetration)])
}

/// Contact between a capsule (`a`) and a cuboid (`b`).
///
/// The capsule's core segment is treated as a swept sphere: the closest point
/// between the segment and the box is located, then the closest point is fed
/// through the sphere-versus-box routine at the capsule radius.
fn capsule_cuboid(
    half_height: f32,
    radius: f32,
    pose_capsule: &Isometry,
    half_extents: Vec3,
    pose_box: &Isometry,
) -> Option<ContactManifold> {
    let obb = Obb::new(half_extents, pose_box);
    let (seg0, seg1) = capsule_segment(half_height, pose_capsule);
    let seg_point = closest_segment_point_to_box(seg0, seg1, &obb);
    let (normal, point_a, point_b, penetration) = point_vs_box(seg_point, radius, &obb)?;
    manifold_from_points(normal, &[(point_a, point_b, penetration)])
}

/// Contact between a capsule (`a`) and a plane half-space (`b`).
///
/// Both cap centers are tested against the plane, so a capsule lying parallel
/// to the plane produces a stable two-point manifold.
fn capsule_plane(
    half_height: f32,
    radius: f32,
    pose_capsule: &Isometry,
    plane_normal: Vec3,
    plane_offset: f32,
    pose_plane: &Isometry,
) -> Option<ContactManifold> {
    let (normal_world, offset_world) = plane_world(plane_normal, plane_offset, pose_plane);
    let (seg0, seg1) = capsule_segment(half_height, pose_capsule);
    let normal = -normal_world;
    let mut points: Vec<(Vec3, Vec3, f32)> = Vec::new();
    for endpoint in [seg0, seg1] {
        let signed = normal_world.dot(endpoint) - offset_world;
        let penetration = radius - signed;
        if penetration >= -CONTACT_TOLERANCE {
            let point_a = endpoint - normal_world * radius;
            let point_b = endpoint - normal_world * signed;
            points.push((point_a, point_b, penetration));
        }
    }
    manifold_from_points(normal, &points)
}

// ---------------------------------------------------------------------------
// Cuboid pairs
// ---------------------------------------------------------------------------

/// Contact between a cuboid (`a`) and a plane half-space (`b`).
///
/// Every box vertex below the plane contributes a contact point; the deepest
/// [`MAX_MANIFOLD_POINTS`] are retained.
fn cuboid_plane(
    half_extents: Vec3,
    pose_box: &Isometry,
    plane_normal: Vec3,
    plane_offset: f32,
    pose_plane: &Isometry,
) -> Option<ContactManifold> {
    let (normal_world, offset_world) = plane_world(plane_normal, plane_offset, pose_plane);
    let obb = Obb::new(half_extents, pose_box);
    let normal = -normal_world;
    let mut points: Vec<(Vec3, Vec3, f32)> = Vec::new();
    for vertex in obb.vertices() {
        let signed = normal_world.dot(vertex) - offset_world;
        let penetration = -signed;
        if penetration >= -CONTACT_TOLERANCE {
            let point_a = vertex;
            let point_b = vertex - normal_world * signed;
            points.push((point_a, point_b, penetration));
        }
    }
    reduce_to_manifold_capacity(&mut points);
    manifold_from_points(normal, &points)
}

/// Contact between two cuboids using the Separating-Axis Test.
///
/// The 15 candidate axes (three face normals per box plus nine edge-edge cross
/// products) are tested. If any axis separates the boxes beyond
/// [`CONTACT_TOLERANCE`], the boxes do not touch and `None` is returned.
/// Otherwise the minimum-penetration axis is selected, with a small bias toward
/// face axes to avoid numerical flip-flopping. A face axis produces a clipped
/// polygonal patch (up to four points); an edge-edge axis produces the single
/// closest-point pair.
fn cuboid_cuboid(
    half_extents_a: Vec3,
    pose_a: &Isometry,
    half_extents_b: Vec3,
    pose_b: &Isometry,
) -> Option<ContactManifold> {
    let a = Obb::new(half_extents_a, pose_a);
    let b = Obb::new(half_extents_b, pose_b);
    let center_delta = b.center - a.center;

    let mut best_face_sep = f32::NEG_INFINITY;
    let mut best_face_axis = Vec3::Y;
    let mut face_ref_is_a = true;
    let mut face_index = 0usize;

    for i in 0..3 {
        let axis = a.axes[i];
        let sep = center_delta.dot(axis).abs() - (a.he[i] + b.support_radius(axis));
        if sep > CONTACT_TOLERANCE {
            return None;
        }
        if sep > best_face_sep {
            best_face_sep = sep;
            best_face_axis = axis;
            face_ref_is_a = true;
            face_index = i;
        }
    }
    for i in 0..3 {
        let axis = b.axes[i];
        let sep = center_delta.dot(axis).abs() - (a.support_radius(axis) + b.he[i]);
        if sep > CONTACT_TOLERANCE {
            return None;
        }
        if sep > best_face_sep {
            best_face_sep = sep;
            best_face_axis = axis;
            face_ref_is_a = false;
            face_index = i;
        }
    }

    let mut best_edge_sep = f32::NEG_INFINITY;
    let mut best_edge_axis = Vec3::Y;
    let mut edge_i = 0usize;
    let mut edge_j = 0usize;
    for i in 0..3 {
        for j in 0..3 {
            let raw = a.axes[i].cross(b.axes[j]);
            let len = raw.length();
            if len < GEOMETRIC_EPS {
                continue;
            }
            let axis = raw / len;
            let sep =
                center_delta.dot(axis).abs() - (a.support_radius(axis) + b.support_radius(axis));
            if sep > CONTACT_TOLERANCE {
                return None;
            }
            if sep > best_edge_sep {
                best_edge_sep = sep;
                best_edge_axis = axis;
                edge_i = i;
                edge_j = j;
            }
        }
    }

    let scale = a.he[0]
        .max(a.he[1])
        .max(a.he[2])
        .max(b.he[0])
        .max(b.he[1])
        .max(b.he[2]);
    let bias = 1.0e-3 * scale + 1.0e-4;

    if best_edge_sep > best_face_sep + bias {
        cuboid_cuboid_edge(
            &a,
            &b,
            center_delta,
            best_edge_axis,
            edge_i,
            edge_j,
            best_edge_sep,
        )
    } else {
        cuboid_cuboid_face(
            &a,
            &b,
            center_delta,
            best_face_axis,
            face_ref_is_a,
            face_index,
        )
    }
}

/// Builds the face-contact patch for the box-box minimum-penetration face axis.
fn cuboid_cuboid_face(
    a: &Obb,
    b: &Obb,
    center_delta: Vec3,
    face_axis: Vec3,
    ref_is_a: bool,
    ref_index: usize,
) -> Option<ContactManifold> {
    // Orient the reference-face normal so it points from the reference box
    // toward the incident box.
    let (reference, incident, ref_normal, manifold_normal) = if ref_is_a {
        let mut n = face_axis;
        if n.dot(center_delta) < 0.0 {
            n = -n;
        }
        (a, b, n, n)
    } else {
        let mut n = face_axis;
        // Reference is B; its outward normal must point toward A (`-center_delta`).
        if n.dot(center_delta) > 0.0 {
            n = -n;
        }
        (b, a, n, -n)
    };

    let (ref_t0, ref_t1) = other_two(ref_index);
    let ref_center = reference.center + ref_normal * reference.he[ref_index];
    let u1 = reference.axes[ref_t0];
    let u2 = reference.axes[ref_t1];
    let e1 = reference.he[ref_t0];
    let e2 = reference.he[ref_t1];

    // Select the incident face: the incident-box face whose outward normal is
    // most anti-parallel to the reference normal.
    let mut best_dot = f32::INFINITY;
    let mut inc_axis = 0usize;
    let mut inc_sign = 1.0f32;
    for k in 0..3 {
        for sign in [1.0f32, -1.0f32] {
            let candidate = incident.axes[k] * sign;
            let d = candidate.dot(ref_normal);
            if d < best_dot {
                best_dot = d;
                inc_axis = k;
                inc_sign = sign;
            }
        }
    }
    let (inc_t0, inc_t1) = other_two(inc_axis);
    let inc_center = incident.center + incident.axes[inc_axis] * (inc_sign * incident.he[inc_axis]);
    let iu1 = incident.axes[inc_t0] * incident.he[inc_t0];
    let iu2 = incident.axes[inc_t1] * incident.he[inc_t1];
    let mut polygon = vec![
        inc_center + iu1 + iu2,
        inc_center + iu1 - iu2,
        inc_center - iu1 - iu2,
        inc_center - iu1 + iu2,
    ];

    // Clip the incident face against the four side planes of the reference face.
    let ref_c1 = ref_center.dot(u1);
    let ref_c2 = ref_center.dot(u2);
    let clipped = clip_polygon(&polygon, u1, ref_c1 + e1);
    let clipped = clip_polygon(&clipped, -u1, -ref_c1 + e1);
    let clipped = clip_polygon(&clipped, u2, ref_c2 + e2);
    let clipped = clip_polygon(&clipped, -u2, -ref_c2 + e2);
    if !clipped.is_empty() {
        polygon = clipped;
    }

    let mut points: Vec<(Vec3, Vec3, f32)> = Vec::new();
    for v in polygon {
        let penetration = -(v - ref_center).dot(ref_normal);
        if penetration >= -CONTACT_TOLERANCE {
            let projected = v + ref_normal * penetration;
            let (point_a, point_b) = if ref_is_a {
                (projected, v)
            } else {
                (v, projected)
            };
            points.push((point_a, point_b, penetration));
        }
    }
    reduce_to_manifold_capacity(&mut points);
    manifold_from_points(manifold_normal, &points)
}

/// Builds the single closest-point contact for a box-box edge-edge axis.
fn cuboid_cuboid_edge(
    a: &Obb,
    b: &Obb,
    center_delta: Vec3,
    axis: Vec3,
    edge_i: usize,
    edge_j: usize,
    separation: f32,
) -> Option<ContactManifold> {
    // Orient the axis so it points from `a` toward `b`.
    let normal = if axis.dot(center_delta) < 0.0 {
        -axis
    } else {
        axis
    };

    let (a_t0, a_t1) = other_two(edge_i);
    let mut mid_a = a.center;
    for j in [a_t0, a_t1] {
        let sign = if a.axes[j].dot(normal) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        mid_a += a.axes[j] * (sign * a.he[j]);
    }
    let a_dir = a.axes[edge_i] * a.he[edge_i];
    let a0 = mid_a + a_dir;
    let a1 = mid_a - a_dir;

    let (b_t0, b_t1) = other_two(edge_j);
    let mut mid_b = b.center;
    for j in [b_t0, b_t1] {
        let sign = if b.axes[j].dot(normal) <= 0.0 {
            1.0
        } else {
            -1.0
        };
        mid_b += b.axes[j] * (sign * b.he[j]);
    }
    let b_dir = b.axes[edge_j] * b.he[edge_j];
    let b0 = mid_b + b_dir;
    let b1 = mid_b - b_dir;

    let (pa, pb) = closest_points_segments(a0, a1, b0, b1);
    let midpoint = (pa + pb) * 0.5;
    let penetration = (-separation).max(0.0);
    let half = penetration * 0.5;
    let point_a = midpoint + normal * half;
    let point_b = midpoint - normal * half;
    manifold_from_points(normal, &[(point_a, point_b, penetration)])
}

// ---------------------------------------------------------------------------
// Shared geometric helpers
// ---------------------------------------------------------------------------

/// An oriented bounding box in world space.
struct Obb {
    /// World-space center of the box.
    center: Vec3,
    /// Orthonormal world-space axes of the box.
    axes: [Vec3; 3],
    /// Half-extents along each corresponding axis.
    he: [f32; 3],
}

impl Obb {
    /// Builds an oriented box from local half-extents and a world pose.
    fn new(half_extents: Vec3, pose: &Isometry) -> Obb {
        let rotation = pose.rotation;
        Obb {
            center: pose.translation,
            axes: [rotation * Vec3::X, rotation * Vec3::Y, rotation * Vec3::Z],
            he: [half_extents.x, half_extents.y, half_extents.z],
        }
    }

    /// Returns the projection radius of the box onto a (unit) `axis`.
    fn support_radius(&self, axis: Vec3) -> f32 {
        self.he[0] * self.axes[0].dot(axis).abs()
            + self.he[1] * self.axes[1].dot(axis).abs()
            + self.he[2] * self.axes[2].dot(axis).abs()
    }

    /// Returns the eight world-space corner vertices of the box.
    fn vertices(&self) -> [Vec3; 8] {
        let x = self.axes[0] * self.he[0];
        let y = self.axes[1] * self.he[1];
        let z = self.axes[2] * self.he[2];
        let c = self.center;
        [
            c + x + y + z,
            c + x + y - z,
            c + x - y + z,
            c + x - y - z,
            c - x + y + z,
            c - x + y - z,
            c - x - y + z,
            c - x - y - z,
        ]
    }
}

/// Returns the two axis indices other than `index` (which is always `0..3`).
fn other_two(index: usize) -> (usize, usize) {
    match index {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Returns the two world-space cap centers of a capsule's core segment.
fn capsule_segment(half_height: f32, pose: &Isometry) -> (Vec3, Vec3) {
    let top = pose.transform_point(Vec3::new(0.0, half_height, 0.0));
    let bottom = pose.transform_point(Vec3::new(0.0, -half_height, 0.0));
    (bottom, top)
}

/// Converts a local plane `dot(normal, x) = offset` into world coordinates.
fn plane_world(normal: Vec3, offset: f32, pose: &Isometry) -> (Vec3, f32) {
    let normal_world = pose.rotation * normal;
    let offset_world = offset + normal_world.dot(pose.translation);
    (normal_world, offset_world)
}

/// Returns the point on segment `[a, b]` closest to `point`.
fn closest_point_on_segment(point: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < GEOMETRIC_EPS {
        return a;
    }
    let t = ((point - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

/// Returns the closest points `(p_on_1, p_on_2)` between two segments.
///
/// This is Ericson's `ClosestPtSegmentSegment` from *Real-Time Collision
/// Detection*, handling degenerate (point-like) segments.
fn closest_points_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    if a < GEOMETRIC_EPS && e < GEOMETRIC_EPS {
        return (p1, p2);
    }

    let (s, t);
    if a < GEOMETRIC_EPS {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e < GEOMETRIC_EPS {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let s_unclamped = if denom > GEOMETRIC_EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t_unclamped = (b * s_unclamped + f) / e;
            if t_unclamped < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t_unclamped > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                t = t_unclamped;
                s = s_unclamped;
            }
        }
    }

    (p1 + d1 * s, p2 + d2 * t)
}

/// Resolves the point/box contact for a sphere or capsule-slice center.
///
/// Returns `(normal, point_a, point_b, penetration)` where the normal points
/// from the point-side shape (`a`) toward the box (`b`), or `None` when the
/// point is farther than `radius` (plus tolerance) from the box.
fn point_vs_box(center: Vec3, radius: f32, obb: &Obb) -> Option<(Vec3, Vec3, Vec3, f32)> {
    let delta = center - obb.center;
    let local = Vec3::new(
        delta.dot(obb.axes[0]),
        delta.dot(obb.axes[1]),
        delta.dot(obb.axes[2]),
    );
    let he = Vec3::new(obb.he[0], obb.he[1], obb.he[2]);
    let clamped = local.clamp(-he, he);
    let inside = local.x.abs() <= he.x && local.y.abs() <= he.y && local.z.abs() <= he.z;

    if !inside {
        let closest = obb.center
            + obb.axes[0] * clamped.x
            + obb.axes[1] * clamped.y
            + obb.axes[2] * clamped.z;
        let diff = closest - center;
        let distance = diff.length();
        let penetration = radius - distance;
        if penetration < -CONTACT_TOLERANCE {
            return None;
        }
        let normal = if distance > GEOMETRIC_EPS {
            diff / distance
        } else {
            fallback_normal()
        };
        let point_a = center + normal * radius;
        let point_b = closest;
        return Some((normal, point_a, point_b, penetration));
    }

    // Center is inside the box: escape along the nearest face.
    let face_pens = [
        he.x - local.x.abs(),
        he.y - local.y.abs(),
        he.z - local.z.abs(),
    ];
    let mut axis_index = 0usize;
    let mut min_pen = face_pens[0];
    if face_pens[1] < min_pen {
        min_pen = face_pens[1];
        axis_index = 1;
    }
    if face_pens[2] < min_pen {
        min_pen = face_pens[2];
        axis_index = 2;
    }
    let local_component = match axis_index {
        0 => local.x,
        1 => local.y,
        _ => local.z,
    };
    let sign = if local_component >= 0.0 { 1.0 } else { -1.0 };
    let outward = obb.axes[axis_index] * sign;
    let normal = -outward;
    let penetration = radius + min_pen;
    let point_b = center + outward * min_pen;
    let point_a = center - outward * radius;
    Some((normal, point_a, point_b, penetration))
}

/// Returns the world-space point on segment `[seg0, seg1]` closest to the box.
///
/// Uses alternating projection between the (convex) segment and (convex) box,
/// which converges to the closest pair; the segment-side point is returned so
/// the caller can run the point-versus-box test at the capsule radius.
fn closest_segment_point_to_box(seg0: Vec3, seg1: Vec3, obb: &Obb) -> Vec3 {
    let to_local = |p: Vec3| {
        let d = p - obb.center;
        Vec3::new(d.dot(obb.axes[0]), d.dot(obb.axes[1]), d.dot(obb.axes[2]))
    };
    let l0 = to_local(seg0);
    let l1 = to_local(seg1);
    let he = Vec3::new(obb.he[0], obb.he[1], obb.he[2]);

    let seg_dir = l1 - l0;
    let len_sq = seg_dir.length_squared();
    let mut t = 0.5f32;
    for _ in 0..24 {
        let seg_point = l0 + seg_dir * t;
        let box_point = seg_point.clamp(-he, he);
        if len_sq < GEOMETRIC_EPS {
            t = 0.0;
            break;
        }
        t = ((box_point - l0).dot(seg_dir) / len_sq).clamp(0.0, 1.0);
    }
    let seg_local = l0 + seg_dir * t;
    obb.center + obb.axes[0] * seg_local.x + obb.axes[1] * seg_local.y + obb.axes[2] * seg_local.z
}

/// Clips a convex polygon against the half-space `dot(v, axis) <= limit`.
///
/// This is one plane of a Sutherland-Hodgman clip; crossing edges are split at
/// the plane boundary.
fn clip_polygon(polygon: &[Vec3], axis: Vec3, limit: f32) -> Vec<Vec3> {
    let mut output: Vec<Vec3> = Vec::new();
    if polygon.is_empty() {
        return output;
    }
    for i in 0..polygon.len() {
        let current = polygon[i];
        let next = polygon[(i + 1) % polygon.len()];
        let dist_current = current.dot(axis) - limit;
        let dist_next = next.dot(axis) - limit;
        if dist_current <= 0.0 {
            output.push(current);
        }
        if (dist_current <= 0.0) != (dist_next <= 0.0) {
            let denom = dist_current - dist_next;
            if denom.abs() > GEOMETRIC_EPS {
                let t = dist_current / denom;
                output.push(current + (next - current) * t);
            }
        }
    }
    output
}

/// Retains only the deepest [`MAX_MANIFOLD_POINTS`] contact points.
fn reduce_to_manifold_capacity(points: &mut Vec<(Vec3, Vec3, f32)>) {
    if points.len() > MAX_MANIFOLD_POINTS {
        points.sort_by(|lhs, rhs| rhs.2.partial_cmp(&lhs.2).unwrap_or(Ordering::Equal));
        points.truncate(MAX_MANIFOLD_POINTS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::ColliderShape;
    use crate::math::transform::Isometry;
    use core::f32::consts::FRAC_PI_2;
    use glam::Quat;

    const EPS: f32 = 1.0e-4;

    fn sphere(radius: f32) -> ColliderShape {
        ColliderShape::Sphere { radius }
    }

    fn cuboid(half: Vec3) -> ColliderShape {
        ColliderShape::Cuboid { half_extents: half }
    }

    fn capsule(half_height: f32, radius: f32) -> ColliderShape {
        ColliderShape::Capsule {
            half_height,
            radius,
        }
    }

    fn plane(normal: Vec3, offset: f32) -> ColliderShape {
        ColliderShape::Plane { normal, offset }
    }

    fn at(translation: Vec3) -> Isometry {
        Isometry::from_translation(translation)
    }

    fn assert_witness_identity(manifold: &ContactManifold) {
        for point in manifold.points() {
            let reconstructed = point.point_b + manifold.normal * point.penetration;
            assert!(
                (reconstructed - point.point_a).length() < 1.0e-3,
                "witness identity violated: a={:?} b={:?} n={:?} pen={}",
                point.point_a,
                point.point_b,
                manifold.normal,
                point.penetration
            );
        }
    }

    #[test]
    fn sphere_sphere_penetration_and_normal() {
        let m = generate_contact(
            &sphere(1.0),
            &at(Vec3::ZERO),
            &sphere(1.0),
            &at(Vec3::X * 1.5),
        )
        .expect("overlapping spheres should contact");
        assert_eq!(m.len(), 1);
        assert!((m.normal - Vec3::X).length() < EPS);
        let p = m.points()[0];
        assert!((p.penetration - 0.5).abs() < EPS);
        assert!((p.point_a - Vec3::new(1.0, 0.0, 0.0)).length() < EPS);
        assert!((p.point_b - Vec3::new(0.5, 0.0, 0.0)).length() < EPS);
        assert_witness_identity(&m);
    }

    #[test]
    fn sphere_sphere_separated_returns_none() {
        assert!(generate_contact(
            &sphere(1.0),
            &at(Vec3::ZERO),
            &sphere(1.0),
            &at(Vec3::X * 3.0)
        )
        .is_none());
    }

    #[test]
    fn sphere_plane_resting() {
        let m = generate_contact(
            &sphere(1.0),
            &at(Vec3::new(0.0, 0.9, 0.0)),
            &plane(Vec3::Y, 0.0),
            &Isometry::IDENTITY,
        )
        .expect("sphere resting on plane should contact");
        assert_eq!(m.len(), 1);
        assert!((m.normal - (-Vec3::Y)).length() < EPS);
        let p = m.points()[0];
        assert!((p.penetration - 0.1).abs() < EPS);
        assert!((p.point_a - Vec3::new(0.0, -0.1, 0.0)).length() < EPS);
        assert!((p.point_b - Vec3::new(0.0, 0.0, 0.0)).length() < EPS);
        assert_witness_identity(&m);
    }

    #[test]
    fn sphere_plane_separated_returns_none() {
        assert!(generate_contact(
            &sphere(1.0),
            &at(Vec3::new(0.0, 2.0, 0.0)),
            &plane(Vec3::Y, 0.0),
            &Isometry::IDENTITY,
        )
        .is_none());
    }

    #[test]
    fn sphere_cuboid_face_contact() {
        let m = generate_contact(
            &sphere(1.0),
            &at(Vec3::new(1.5, 0.0, 0.0)),
            &cuboid(Vec3::splat(1.0)),
            &Isometry::IDENTITY,
        )
        .expect("sphere overlapping box should contact");
        assert_eq!(m.len(), 1);
        assert!((m.normal - (-Vec3::X)).length() < EPS);
        let p = m.points()[0];
        assert!((p.penetration - 0.5).abs() < EPS);
        assert!((p.point_a - Vec3::new(0.5, 0.0, 0.0)).length() < EPS);
        assert!((p.point_b - Vec3::new(1.0, 0.0, 0.0)).length() < EPS);
        assert_witness_identity(&m);
    }

    #[test]
    fn cuboid_sphere_swapped_flips_normal() {
        let m = generate_contact(
            &cuboid(Vec3::splat(1.0)),
            &Isometry::IDENTITY,
            &sphere(1.0),
            &at(Vec3::new(1.5, 0.0, 0.0)),
        )
        .expect("box overlapping sphere should contact");
        assert_eq!(m.len(), 1);
        // Box is `a`, sphere is `b`; normal points from box toward sphere (+X).
        assert!((m.normal - Vec3::X).length() < EPS);
        assert_witness_identity(&m);
    }

    #[test]
    fn sphere_inside_cuboid() {
        let m = generate_contact(
            &sphere(0.5),
            &at(Vec3::ZERO),
            &cuboid(Vec3::splat(2.0)),
            &Isometry::IDENTITY,
        )
        .expect("sphere inside box should contact");
        assert_eq!(m.len(), 1);
        let p = m.points()[0];
        assert!((p.penetration - 2.5).abs() < EPS);
        assert!((m.normal.length() - 1.0).abs() < EPS);
        assert_witness_identity(&m);
    }

    #[test]
    fn sphere_capsule_contact() {
        let m = generate_contact(
            &sphere(0.5),
            &at(Vec3::new(0.8, 0.0, 0.0)),
            &capsule(1.0, 0.5),
            &Isometry::IDENTITY,
        )
        .expect("sphere touching capsule should contact");
        assert_eq!(m.len(), 1);
        assert!((m.normal - (-Vec3::X)).length() < EPS);
        let p = m.points()[0];
        assert!((p.penetration - 0.2).abs() < EPS);
        assert_witness_identity(&m);
    }

    #[test]
    fn capsule_capsule_crossing() {
        // Capsule A along Y at origin; capsule B along X offset in +Z.
        let pose_b = Isometry::new(Vec3::new(0.0, 0.0, 0.4), Quat::from_rotation_z(FRAC_PI_2));
        let m = generate_contact(
            &capsule(1.0, 0.5),
            &Isometry::IDENTITY,
            &capsule(1.0, 0.5),
            &pose_b,
        )
        .expect("crossing capsules should contact");
        assert_eq!(m.len(), 1);
        assert!((m.normal - Vec3::Z).length() < EPS);
        let p = m.points()[0];
        assert!((p.penetration - 0.6).abs() < EPS);
        assert_witness_identity(&m);
    }

    #[test]
    fn capsule_capsule_separated_returns_none() {
        let pose_b = Isometry::new(Vec3::new(0.0, 0.0, 2.0), Quat::from_rotation_z(FRAC_PI_2));
        assert!(generate_contact(
            &capsule(1.0, 0.5),
            &Isometry::IDENTITY,
            &capsule(1.0, 0.5),
            &pose_b
        )
        .is_none());
    }

    #[test]
    fn capsule_plane_two_points() {
        // Horizontal capsule (axis along X) resting above a Y-up plane.
        let pose = Isometry::new(Vec3::new(0.0, 0.4, 0.0), Quat::from_rotation_z(FRAC_PI_2));
        let m = generate_contact(
            &capsule(1.0, 0.5),
            &pose,
            &plane(Vec3::Y, 0.0),
            &Isometry::IDENTITY,
        )
        .expect("horizontal capsule on plane should contact");
        assert_eq!(m.len(), 2);
        assert!((m.normal - (-Vec3::Y)).length() < EPS);
        for p in m.points() {
            assert!((p.penetration - 0.1).abs() < EPS);
        }
        assert_witness_identity(&m);
    }

    #[test]
    fn capsule_cuboid_side_contact() {
        // Vertical capsule beside an axis-aligned box.
        let m = generate_contact(
            &capsule(1.0, 0.5),
            &at(Vec3::new(1.4, 0.0, 0.0)),
            &cuboid(Vec3::splat(1.0)),
            &Isometry::IDENTITY,
        )
        .expect("capsule beside box should contact");
        assert!(!m.is_empty());
        assert!((m.normal - (-Vec3::X)).length() < EPS);
        let p = m.points()[0];
        assert!((p.penetration - 0.1).abs() < EPS);
        assert_witness_identity(&m);
    }

    #[test]
    fn cuboid_plane_resting_four_points() {
        let m = generate_contact(
            &cuboid(Vec3::splat(1.0)),
            &at(Vec3::new(0.0, 0.9, 0.0)),
            &plane(Vec3::Y, 0.0),
            &Isometry::IDENTITY,
        )
        .expect("box resting on plane should contact");
        assert_eq!(m.len(), 4);
        assert!((m.normal - (-Vec3::Y)).length() < EPS);
        for p in m.points() {
            assert!((p.penetration - 0.1).abs() < EPS);
        }
        assert_witness_identity(&m);
    }

    #[test]
    fn cuboid_cuboid_resting_face_face_four_points() {
        // Box A at origin (spans y in [-1, 1]); box B stacked on top with a
        // small overlap.
        let m = generate_contact(
            &cuboid(Vec3::splat(1.0)),
            &at(Vec3::ZERO),
            &cuboid(Vec3::splat(1.0)),
            &at(Vec3::new(0.0, 1.9, 0.0)),
        )
        .expect("stacked boxes should contact");
        assert_eq!(m.len(), 4);
        assert!((m.normal - Vec3::Y).length() < EPS);
        for p in m.points() {
            assert!((p.penetration - 0.1).abs() < EPS);
        }
        assert_witness_identity(&m);
    }

    #[test]
    fn cuboid_cuboid_corner_poke_fewer_points() {
        // A small box, tilted about two axes so a corner points downward, poking
        // into the top face of the lower box.
        let rotation = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(0.6);
        let pose_b = Isometry::new(Vec3::new(0.0, 1.7, 0.0), rotation);
        let m = generate_contact(
            &cuboid(Vec3::splat(1.0)),
            &at(Vec3::ZERO),
            &cuboid(Vec3::splat(0.5)),
            &pose_b,
        )
        .expect("corner poke should contact");
        assert!(!m.is_empty());
        assert!(m.len() < 4);
        assert!(m.normal.y > 0.5);
        assert_witness_identity(&m);
    }

    #[test]
    fn cuboid_cuboid_separated_returns_none() {
        assert!(generate_contact(
            &cuboid(Vec3::splat(1.0)),
            &at(Vec3::ZERO),
            &cuboid(Vec3::splat(1.0)),
            &at(Vec3::new(0.0, 3.0, 0.0)),
        )
        .is_none());
    }

    #[test]
    fn plane_plane_returns_none() {
        assert!(generate_contact(
            &plane(Vec3::Y, 0.0),
            &Isometry::IDENTITY,
            &plane(Vec3::Y, 1.0),
            &Isometry::IDENTITY,
        )
        .is_none());
    }
}
