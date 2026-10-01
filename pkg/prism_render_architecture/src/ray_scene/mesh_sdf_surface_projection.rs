//! Projection of an arbitrary point onto the nearest surface of a signed
//! distance field for the `CPU` golden path.
//!
//! A signed distance field answers two questions at every point: how far the
//! nearest surface is (the magnitude) and which way it lies (the gradient).
//! Combining them lets us *snap* a point onto the surface by repeatedly
//! stepping along the inward/outward normal by the signed distance — a
//! Newton-style root find on the field. `AAA` engines use this to resolve
//! penetrations in distance-field collisions, stick decals and particles to
//! geometry, and seed contact points for soft-body and cloth solvers.
//!
//! This module composes the trilinear sampler
//! ([`super::mesh_sdf_raymarch::sample_signed_distance`]) with the gradient
//! normal ([`super::mesh_sdf_normal::sdf_normal`]). Each iteration moves the
//! point by `-normal * signed_distance`, which drives the signed distance
//! toward zero from either side. It is transcendental-free apart from the
//! `sqrt` the normal already uses, so the result is reproducible.

use super::mesh_sdf_normal::sdf_normal;
use super::mesh_sdf_raymarch::sample_signed_distance;
use super::mesh_signed_distance_field::SignedDistanceField;

/// Result of projecting a point onto the zero level set of a signed distance
/// field.
///
/// Returned by [`project_to_surface`]. The point is the converged (or
/// best-effort) location, the residual is its remaining unsigned distance to
/// the surface, and the iteration count reports how many Newton steps were
/// taken (zero when the input already lay on the surface).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceProjection {
    /// World-space position after projection.
    point: [f32; 3],
    /// Remaining unsigned distance to the surface at [`SurfaceProjection::point`].
    residual: f32,
    /// Number of Newton steps taken to reach the result.
    iterations: u32,
}

impl SurfaceProjection {
    /// World-space position after projection.
    pub fn point(&self) -> [f32; 3] {
        self.point
    }

    /// Remaining unsigned distance to the surface.
    pub fn residual(&self) -> f32 {
        self.residual
    }

    /// Number of Newton steps taken to reach the result.
    pub fn iterations(&self) -> u32 {
        self.iterations
    }
}

/// Projects `point` onto the nearest surface of the field by iterating the
/// Newton step `point -= normal * signed_distance`.
///
/// Iteration stops as soon as the unsigned signed distance drops to or below
/// `tolerance`, or after `max_iterations` steps. Returns `None` when the field
/// gradient vanishes before convergence (a flat, constant region offers no
/// direction to step), otherwise the best-effort [`SurfaceProjection`].
pub fn project_to_surface(
    field: &SignedDistanceField,
    point: [f32; 3],
    max_iterations: u32,
    tolerance: f32,
) -> Option<SurfaceProjection> {
    let mut current = point;
    for iteration in 0..max_iterations {
        let signed = sample_signed_distance(field, current);
        if signed.abs() <= tolerance {
            return Some(SurfaceProjection {
                point: current,
                residual: signed.abs(),
                iterations: iteration,
            });
        }
        let normal = sdf_normal(field, current)?;
        current = [
            current[0] - normal[0] * signed,
            current[1] - normal[1] * signed,
            current[2] - normal[2] * signed,
        ];
    }
    let residual = sample_signed_distance(field, current).abs();
    Some(SurfaceProjection {
        point: current,
        residual,
        iterations: max_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::mesh_signed_distance_field::signed_distance_field;
    use crate::ray_scene::mesh_voxelize::voxelize_surface;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh from positions (no normals/uvs).
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// Closed axis-aligned unit cube (12 triangles).
    fn cube() -> TriangleMesh {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let i = vec![
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ];
        mesh(p, i)
    }

    /// Signed distance field of the unit cube at resolution 4.
    fn cube_field() -> SignedDistanceField {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        signed_distance_field(&grid)
    }

    /// World-space point at fractional position `(fx, fy, fz)` of the grid.
    fn grid_point(field: &SignedDistanceField, fx: f32, fy: f32, fz: f32) -> [f32; 3] {
        let dims = field.dims();
        let voxel = field.voxel_size();
        let origin = field.origin();
        [
            origin[0] + fx * dims[0] as f32 * voxel,
            origin[1] + fy * dims[1] as f32 * voxel,
            origin[2] + fz * dims[2] as f32 * voxel,
        ]
    }

    #[test]
    fn converges_within_a_voxel() {
        let field = cube_field();
        // An asymmetric interior point has a well-defined gradient to follow.
        let point = grid_point(&field, 0.45, 0.4, 0.42);
        let tolerance = field.voxel_size() * 1.0;
        let projection = project_to_surface(&field, point, 64, tolerance)
            .expect("projection must orient");
        assert!(projection.residual() <= tolerance);
    }

    #[test]
    fn interior_point_projects_onto_surface() {
        let field = cube_field();
        let voxel = field.voxel_size();
        let point = grid_point(&field, 0.45, 0.4, 0.42);
        let tolerance = 0.5 * voxel;
        let projection = project_to_surface(&field, point, 64, tolerance)
            .expect("interior projection must orient");
        // The projection drives the residual below the tolerance.
        assert!(
            projection.residual() <= tolerance,
            "residual {} exceeds tolerance {tolerance}",
            projection.residual(),
        );
        // And the result actually sits near the zero level set.
        let sampled = sample_signed_distance(&field, projection.point());
        assert!(sampled.abs() <= tolerance, "sampled {sampled}");
    }

    #[test]
    fn projection_reduces_distance() {
        let field = cube_field();
        let point = grid_point(&field, 0.45, 0.4, 0.42);
        let before = sample_signed_distance(&field, point).abs();
        let projection = project_to_surface(&field, point, 64, 1e-4)
            .expect("projection must orient");
        assert!(
            projection.residual() < before,
            "residual {} should be below the initial {before}",
            projection.residual(),
        );
    }

    #[test]
    fn already_on_surface_takes_zero_iterations() {
        let field = cube_field();
        // Any point already within tolerance reports zero Newton steps.
        let point = grid_point(&field, 0.45, 0.4, 0.42);
        let huge_tolerance = 10.0;
        let projection = project_to_surface(&field, point, 32, huge_tolerance)
            .expect("projection must orient");
        assert_eq!(projection.iterations(), 0);
        assert_eq!(projection.point(), point);
    }

    #[test]
    fn degenerate_critical_point_cannot_project() {
        let field = cube_field();
        // The exact cube center is a symmetric minimum of the field: the
        // gradient vanishes there, so no projection direction exists even
        // though the point is well inside the solid.
        let center = grid_point(&field, 0.5, 0.5, 0.5);
        assert!(sample_signed_distance(&field, center) < 0.0);
        assert!(project_to_surface(&field, center, 32, 1e-4).is_none());
    }

    #[test]
    fn projected_point_moves_from_input() {
        let field = cube_field();
        let input = grid_point(&field, 0.45, 0.4, 0.42);
        let projection = project_to_surface(&field, input, 64, 1e-4)
            .expect("projection must orient");
        let moved = projection.point();
        let delta_squared = (moved[0] - input[0]) * (moved[0] - input[0])
            + (moved[1] - input[1]) * (moved[1] - input[1])
            + (moved[2] - input[2]) * (moved[2] - input[2]);
        assert!(delta_squared > 0.0, "projection should displace the input");
    }
}
