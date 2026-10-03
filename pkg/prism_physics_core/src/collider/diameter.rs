//! Exact diameter (farthest-pair distance) of a point set or mesh.
//!
//! The diameter of a shape is the greatest distance between any two of its
//! points. It bounds how far apart two features can be and feeds broad-phase
//! sizing, LOD switch distances, and sanity checks on authored assets. A
//! classical result guarantees the farthest pair always lies on the convex
//! hull, so the diameter of a cloud equals the diameter of its hull vertices.
//! This module computes the convex hull once
//! ([`convex_hull`](crate::collider::convex_hull)) and then takes the exact
//! maximum over hull-vertex pairs, which is cheap because the hull is tiny
//! relative to the input. When a hull cannot be formed (fewer than four
//! non-coplanar points) it falls back to an exact brute-force scan of the raw
//! input, so the answer is always exact rather than approximate.
//!
//! Unlike [`minimal_bounding_sphere`](crate::collider::minimal_bounding_sphere),
//! whose diameter is an upper bound, this returns the true farthest-pair
//! distance and the two endpoints that realise it.
//!
//! This is pure point-cloud geometry with no coupling to the collision
//! pipeline, and nothing here is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::hull::convex_hull;

/// The farthest-pair distance of a point set and the two points realising it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshDiameter {
    /// The farthest-pair (diameter) distance.
    pub diameter: f32,
    /// One endpoint of the farthest pair.
    pub endpoint_a: Vec3,
    /// The other endpoint of the farthest pair.
    pub endpoint_b: Vec3,
}

impl MeshDiameter {
    /// Half the diameter: a lower bound on any enclosing-sphere radius.
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.diameter * 0.5
    }

    /// The midpoint of the farthest pair.
    #[must_use]
    pub fn midpoint(&self) -> Vec3 {
        (self.endpoint_a + self.endpoint_b) * 0.5
    }
}

/// Computes the exact diameter (farthest-pair distance) of a point set.
///
/// Reduces to the convex-hull vertices when a hull can be formed, then takes
/// the exact maximum over hull-vertex pairs; otherwise it scans the raw input
/// exactly. Returns `None` when fewer than two points are supplied or when no
/// pair has a finite positive separation (for example a single repeated point).
#[must_use]
pub fn mesh_diameter(points: &[Vec3]) -> Option<MeshDiameter> {
    if points.len() < 2 {
        return None;
    }

    // The farthest pair lies on the convex hull, so reduce to hull vertices
    // when a hull exists; otherwise fall back to the raw (near-degenerate)
    // input so the scan is still exact.
    let candidates: &[Vec3] = match convex_hull(points) {
        Some((ref hull_vertices, _)) if hull_vertices.len() >= 2 => {
            return farthest_pair(hull_vertices);
        }
        _ => points,
    };

    farthest_pair(candidates)
}

/// Exact O(n^2) farthest-pair scan over a point slice.
fn farthest_pair(points: &[Vec3]) -> Option<MeshDiameter> {
    let mut best = -1.0_f32;
    let mut best_a = 0_usize;
    let mut best_b = 0_usize;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let d = (points[i] - points[j]).length_squared();
            if d > best {
                best = d;
                best_a = i;
                best_b = j;
            }
        }
    }

    if best <= 0.0 {
        return None;
    }

    Some(MeshDiameter {
        diameter: best.sqrt(),
        endpoint_a: points[best_a],
        endpoint_b: points[best_b],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fewer_than_two_points_is_rejected() {
        assert!(mesh_diameter(&[]).is_none());
        assert!(mesh_diameter(&[Vec3::ZERO]).is_none());
    }

    #[test]
    fn repeated_point_has_no_diameter() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert!(mesh_diameter(&[p, p, p]).is_none());
    }

    #[test]
    fn two_points_diameter_is_their_distance() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let d = mesh_diameter(&[a, b]).expect("two distinct points");
        assert!(
            (d.diameter - 3.0).abs() < 1.0e-6,
            "diameter = {}",
            d.diameter
        );
        assert!((d.midpoint() - Vec3::new(0.5, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn unit_cube_diameter_is_body_diagonal() {
        let mut verts = Vec::new();
        for x in [-1.0_f32, 1.0] {
            for y in [-1.0_f32, 1.0] {
                for z in [-1.0_f32, 1.0] {
                    verts.push(Vec3::new(x, y, z));
                }
            }
        }
        let d = mesh_diameter(&verts).expect("cube has a diameter");
        // Body diagonal of a 2x2x2 cube is 2*sqrt(3).
        let expected = 2.0 * (3.0_f32).sqrt();
        assert!(
            (d.diameter - expected).abs() < 1.0e-4,
            "diameter = {}, expected {}",
            d.diameter,
            expected
        );
        // Endpoints must be opposite corners (midpoint at the centre).
        assert!(
            d.midpoint().length() < 1.0e-4,
            "midpoint = {:?}",
            d.midpoint()
        );
    }

    #[test]
    fn interior_points_do_not_change_diameter() {
        let mut verts = vec![
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        // A swarm of interior points must not inflate the diameter.
        for i in 0..20 {
            let t = i as f32 * 0.05 - 0.5;
            verts.push(Vec3::new(t, t, t));
        }
        let d = mesh_diameter(&verts).expect("has a diameter");
        assert!(
            (d.diameter - 10.0).abs() < 1.0e-4,
            "diameter = {}",
            d.diameter
        );
    }

    #[test]
    fn coplanar_points_fall_back_to_exact_scan() {
        // A flat square in the z=0 plane cannot form a 3D hull, exercising the
        // brute-force fallback; the diameter is the square's diagonal.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let d = mesh_diameter(&verts).expect("coplanar square has a diameter");
        let expected = (2.0_f32).sqrt();
        assert!(
            (d.diameter - expected).abs() < 1.0e-4,
            "diameter = {}, expected {}",
            d.diameter,
            expected
        );
    }
}
