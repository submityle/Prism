//! Surface normals of a signed distance field by gradient estimation for the
//! `CPU` golden path.
//!
//! The gradient of a signed distance field points from the solid interior
//! (negative) toward the exterior (positive), so — once normalized — it is the
//! outward surface normal at any sampled point. `AAA` renderers need this
//! normal to shade sphere-traced hits, build distance-field soft shadows, and
//! bend ambient-occlusion and global-illumination cones around geometry.
//!
//! This module is the shading-side companion of
//! [`super::mesh_sdf_raymarch`]: it differentiates the same trilinear sampler
//! ([`super::mesh_sdf_raymarch::sample_signed_distance`]) with a symmetric
//! central difference. Everything stays transcendental-free — the only
//! non-linear operation is a single `sqrt` to normalize the gradient — so the
//! result is reproducible across machines.

use super::mesh_sdf_raymarch::sample_signed_distance;
use super::mesh_signed_distance_field::SignedDistanceField;

/// Estimates the gradient of the signed distance field at `point` with a
/// symmetric central difference.
///
/// Each component is the difference of the field sampled one half-step on
/// either side of `point` along that axis, divided by the step. The step is a
/// single voxel edge, matching the field's native resolution. The resulting
/// vector points outward (toward increasing signed distance) with a magnitude
/// near one for a well-formed field; it is not normalized here.
pub fn sdf_gradient(field: &SignedDistanceField, point: [f32; 3]) -> [f32; 3] {
    let step = field.voxel_size();
    let inv = 1.0 / (2.0 * step);
    let mut gradient = [0f32; 3];
    for (axis, slot) in gradient.iter_mut().enumerate() {
        let mut forward = point;
        let mut backward = point;
        forward[axis] += step;
        backward[axis] -= step;
        let difference =
            sample_signed_distance(field, forward) - sample_signed_distance(field, backward);
        *slot = difference * inv;
    }
    gradient
}

/// Returns the outward unit surface normal at `point` (the normalized
/// [`sdf_gradient`]), or `None` when the gradient is too short to orient
/// reliably (for example in a flat, constant region of the field).
pub fn sdf_normal(field: &SignedDistanceField, point: [f32; 3]) -> Option<[f32; 3]> {
    let gradient = sdf_gradient(field, point);
    let length_squared =
        gradient[0] * gradient[0] + gradient[1] * gradient[1] + gradient[2] * gradient[2];
    if length_squared <= f32::MIN_POSITIVE {
        return None;
    }
    let length = length_squared.sqrt();
    Some([
        gradient[0] / length,
        gradient[1] / length,
        gradient[2] / length,
    ])
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
    fn gradient_points_outward_near_positive_x_face() {
        let field = cube_field();
        let point = grid_point(&field, 0.8, 0.5, 0.5);
        let gradient = sdf_gradient(&field, point);
        // Outward near the +x face means a positive, dominant x component.
        assert!(gradient[0] > 0.0, "x gradient must be positive: {gradient:?}");
        assert!(gradient[0].abs() > gradient[1].abs());
        assert!(gradient[0].abs() > gradient[2].abs());
    }

    #[test]
    fn gradient_flips_across_opposite_faces() {
        let field = cube_field();
        let near_max = sdf_gradient(&field, grid_point(&field, 0.8, 0.5, 0.5));
        let near_min = sdf_gradient(&field, grid_point(&field, 0.2, 0.5, 0.5));
        // The outward x direction reverses between the +x and -x faces.
        assert!(near_max[0] > 0.0);
        assert!(near_min[0] < 0.0);
    }

    #[test]
    fn normal_is_unit_length() {
        let field = cube_field();
        let normal = sdf_normal(&field, grid_point(&field, 0.8, 0.5, 0.5))
            .expect("gradient near a face must orient");
        let length =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!((length - 1.0).abs() <= 1e-5, "normal length {length}");
    }

    #[test]
    fn normal_points_outward_near_negative_z_face() {
        let field = cube_field();
        let normal = sdf_normal(&field, grid_point(&field, 0.5, 0.5, 0.2))
            .expect("gradient near a face must orient");
        // Outward near the -z face means a negative, dominant z component.
        assert!(normal[2] < 0.0, "z normal must be negative: {normal:?}");
        assert!(normal[2].abs() > normal[0].abs());
        assert!(normal[2].abs() > normal[1].abs());
    }

    #[test]
    fn gradient_is_symmetric_central_difference() {
        // A symmetric difference about the same point is odd: sampling the
        // reversed offsets negates every component.
        let field = cube_field();
        let point = grid_point(&field, 0.6, 0.4, 0.7);
        let gradient = sdf_gradient(&field, point);
        // Reconstruct manually to confirm the central-difference definition.
        let step = field.voxel_size();
        let inv = 1.0 / (2.0 * step);
        for axis in 0..3 {
            let mut fwd = point;
            let mut bwd = point;
            fwd[axis] += step;
            bwd[axis] -= step;
            let expected =
                (sample_signed_distance(&field, fwd) - sample_signed_distance(&field, bwd)) * inv;
            assert!((gradient[axis] - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn constant_region_has_no_orientation() {
        // Far outside the grid every sample clamps to the same border shell, so
        // the central difference vanishes and no normal can be derived.
        let field = cube_field();
        let origin = field.origin();
        let far = [origin[0] - 50.0, origin[1] - 50.0, origin[2] - 50.0];
        assert!(
            sdf_normal(&field, far).is_none(),
            "a constant clamped region must not yield a normal",
        );
    }
}
