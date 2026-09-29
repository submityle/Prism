//! Convex polyhedra represented by vertices and outward-oriented face loops.
//!
//! A [`ConvexPolyhedron`] is the workhorse of the fracture module: fracture
//! cells are built as the intersection of half-spaces (the six planes of the
//! bounding shape plus the Voronoi bisectors to neighbouring sites) via
//! [`ConvexPolyhedron::from_halfspaces`], then measured with
//! [`ConvexPolyhedron::volume`] / [`ConvexPolyhedron::centroid`].
//!
//! Construction uses *vertex enumeration*: every triple of planes is
//! intersected, and a candidate point is kept only when it lies inside every
//! half-space. Surviving points are the polytope vertices; grouping them by the
//! plane they lie on and ordering each group around the face normal yields the
//! face loops. This is robust for the small plane counts a single cell needs
//! and avoids the connectivity bookkeeping of incremental plane clipping.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Vertex
//! enumeration of an H-representation polytope and the divergence-theorem
//! volume/centroid integrals are standard, publicly documented computational-
//! geometry results.

use glam::Vec3;

use crate::fracture::plane::{intersect_three, Plane};
use crate::fracture::predicates::{points_close, pseudo_angle};
use crate::math::scalar::Real;

/// Relative tolerance (times the polytope scale) for deciding a candidate
/// vertex satisfies every half-space.
const FEASIBLE_REL_EPS: Real = 1e-3;
/// Relative tolerance for deciding a vertex lies on a given face plane.
const ON_PLANE_REL_EPS: Real = 1e-3;
/// Relative tolerance for merging duplicate vertices.
const MERGE_REL_EPS: Real = 1e-4;
/// Absolute determinant floor for accepting a three-plane intersection.
const DET_EPS: Real = 1e-6;

/// A bounded convex polyhedron given by its vertices and its face loops.
///
/// Each entry of [`ConvexPolyhedron::faces`] is a list of indices into
/// [`ConvexPolyhedron::vertices`] wound counter-clockwise when viewed from
/// outside the solid, so face normals point outward.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvexPolyhedron {
    vertices: Vec<Vec3>,
    faces: Vec<Vec<usize>>,
}

impl ConvexPolyhedron {
    /// Builds the axis-aligned box spanning `min..=max` as a closed convex
    /// polyhedron with eight vertices and six outward faces.
    #[must_use]
    pub fn box_aabb(min: Vec3, max: Vec3) -> ConvexPolyhedron {
        let vertices = vec![
            Vec3::new(min.x, min.y, min.z), // 0
            Vec3::new(max.x, min.y, min.z), // 1
            Vec3::new(max.x, max.y, min.z), // 2
            Vec3::new(min.x, max.y, min.z), // 3
            Vec3::new(min.x, min.y, max.z), // 4
            Vec3::new(max.x, min.y, max.z), // 5
            Vec3::new(max.x, max.y, max.z), // 6
            Vec3::new(min.x, max.y, max.z), // 7
        ];
        // Each loop is CCW seen from outside (normal points away from centre).
        let faces = vec![
            vec![0, 3, 2, 1], // -Z
            vec![4, 5, 6, 7], // +Z
            vec![0, 1, 5, 4], // -Y
            vec![3, 7, 6, 2], // +Y
            vec![0, 4, 7, 3], // -X
            vec![1, 2, 6, 5], // +X
        ];
        ConvexPolyhedron { vertices, faces }
    }

    /// Builds the convex polyhedron equal to the intersection of the interior
    /// half-spaces (`n · x <= offset`) of `planes`, or returns `None` when the
    /// intersection is empty or degenerate (fewer than four vertices/faces).
    #[must_use]
    pub fn from_halfspaces(planes: &[Plane]) -> Option<ConvexPolyhedron> {
        if planes.len() < 4 {
            return None;
        }
        let scale = plane_scale(planes);
        let feasible_eps = FEASIBLE_REL_EPS * scale;
        let merge_eps = MERGE_REL_EPS * scale;
        let on_eps = ON_PLANE_REL_EPS * scale;

        // Vertex enumeration over every triple of planes.
        let mut vertices: Vec<Vec3> = Vec::new();
        let n = planes.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let Some(p) = intersect_three(&planes[i], &planes[j], &planes[k], DET_EPS)
                    else {
                        continue;
                    };
                    if !planes.iter().all(|pl| pl.contains(p, feasible_eps)) {
                        continue;
                    }
                    if vertices.iter().any(|v| points_close(*v, p, merge_eps)) {
                        continue;
                    }
                    vertices.push(p);
                }
            }
        }
        if vertices.len() < 4 {
            return None;
        }

        // Group vertices onto each plane and order them into a face loop.
        let mut faces: Vec<Vec<usize>> = Vec::new();
        for plane in planes {
            let mut on: Vec<usize> = (0..vertices.len())
                .filter(|&vi| plane.signed_distance(vertices[vi]).abs() <= on_eps)
                .collect();
            if on.len() < 3 {
                continue;
            }
            order_face_loop(&vertices, &mut on, plane.normal);
            faces.push(on);
        }
        if faces.len() < 4 {
            return None;
        }
        Some(ConvexPolyhedron { vertices, faces })
    }

    /// Returns the polyhedron vertices.
    #[must_use]
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// Returns the face loops as index lists into [`ConvexPolyhedron::vertices`].
    #[must_use]
    pub fn faces(&self) -> &[Vec<usize>] {
        &self.faces
    }

    /// Returns the outward face planes, one per face, derived from the first
    /// three vertices of each loop.
    #[must_use]
    pub fn face_planes(&self) -> Vec<Plane> {
        let mut out = Vec::with_capacity(self.faces.len());
        for face in &self.faces {
            let a = self.vertices[face[0]];
            let b = self.vertices[face[1]];
            let c = self.vertices[face[2]];
            let normal = (b - a).cross(c - a);
            out.push(Plane::from_point_normal(a, normal));
        }
        out
    }

    /// Emits the outward-oriented triangle fan of every face (CCW seen from
    /// outside), used by the divergence-theorem integrals.
    #[must_use]
    pub fn face_triangles(&self) -> Vec<[Vec3; 3]> {
        let mut tris = Vec::new();
        for face in &self.faces {
            if face.len() < 3 {
                continue;
            }
            let a = self.vertices[face[0]];
            for w in 1..(face.len() - 1) {
                let b = self.vertices[face[w]];
                let c = self.vertices[face[w + 1]];
                tris.push([a, b, c]);
            }
        }
        tris
    }

    /// Returns the enclosed volume via the divergence theorem.
    #[must_use]
    pub fn volume(&self) -> Real {
        let mut vol6 = 0.0;
        for [a, b, c] in self.face_triangles() {
            vol6 += a.dot(b.cross(c));
        }
        (vol6 / 6.0).abs()
    }

    /// Returns the volumetric centroid (centre of mass of a uniform-density
    /// solid). Falls back to the vertex average for a degenerate zero-volume
    /// polyhedron.
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        let mut vol6 = 0.0;
        let mut acc = Vec3::ZERO;
        for [a, b, c] in self.face_triangles() {
            let d = a.dot(b.cross(c));
            vol6 += d;
            acc += d * (a + b + c);
        }
        if vol6.abs() <= Real::EPSILON {
            return self.vertex_average();
        }
        acc / (4.0 * vol6)
    }

    /// Returns `true` when `p` lies inside every face's interior half-space
    /// within `eps` tolerance.
    #[must_use]
    pub fn contains(&self, p: Vec3, eps: Real) -> bool {
        self.face_planes().iter().all(|pl| pl.contains(p, eps))
    }

    /// Arithmetic mean of the vertices (a convex-interior fallback point).
    #[must_use]
    fn vertex_average(&self) -> Vec3 {
        if self.vertices.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = self.vertices.iter().copied().sum();
        sum / (self.vertices.len() as Real)
    }
}

/// Characteristic scale of a plane set, used to make tolerances scale-aware.
fn plane_scale(planes: &[Plane]) -> Real {
    let mut s: Real = 1.0;
    for p in planes {
        s = s.max(p.offset.abs());
    }
    s
}

/// Orders the vertex indices `loop_ids` into a convex face loop wound CCW
/// around `normal`, in place.
fn order_face_loop(vertices: &[Vec3], loop_ids: &mut [usize], normal: Vec3) {
    let centre: Vec3 =
        loop_ids.iter().map(|&i| vertices[i]).sum::<Vec3>() / (loop_ids.len() as Real);
    // Build an in-plane orthonormal basis (u, v) with (u, v, normal) right-handed.
    let reference = if normal.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let u = normal.cross(reference).normalize_or_zero();
    let u = if u == Vec3::ZERO { Vec3::X } else { u };
    let v = normal.cross(u);
    loop_ids.sort_by(|&ia, &ib| {
        let da = vertices[ia] - centre;
        let db = vertices[ib] - centre;
        let ka = pseudo_angle(da.dot(u), da.dot(v));
        let kb = pseudo_angle(db.dot(u), db.dot(v));
        ka.partial_cmp(&kb).unwrap_or(core::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_volume_and_centroid() {
        let b = ConvexPolyhedron::box_aabb(Vec3::ZERO, Vec3::splat(2.0));
        assert!((b.volume() - 8.0).abs() < 1e-4);
        assert!((b.centroid() - Vec3::splat(1.0)).length() < 1e-4);
    }

    #[test]
    fn halfspace_intersection_rebuilds_box() {
        let planes = ConvexPolyhedron::box_aabb(Vec3::ZERO, Vec3::ONE).face_planes();
        let rebuilt = ConvexPolyhedron::from_halfspaces(&planes).expect("non-empty");
        assert!((rebuilt.volume() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn half_clip_gives_half_volume() {
        let mut planes = ConvexPolyhedron::box_aabb(Vec3::ZERO, Vec3::ONE).face_planes();
        // Cut through the centre along +X, keeping x <= 0.5.
        planes.push(Plane::from_point_normal(Vec3::new(0.5, 0.5, 0.5), Vec3::X));
        let clipped = ConvexPolyhedron::from_halfspaces(&planes).expect("non-empty");
        assert!((clipped.volume() - 0.5).abs() < 1e-3);
    }
}
