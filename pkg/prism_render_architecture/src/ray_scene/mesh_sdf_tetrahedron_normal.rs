//! Surface normals of a signed distance field by the tetrahedron technique for
//! the `CPU` golden path.
//!
//! The symmetric central difference in [`super::mesh_sdf_normal::sdf_gradient`]
//! needs six field samples (two per axis) to estimate the gradient. The
//! *tetrahedron technique* (popularized by Inigo Quilez) reaches the same
//! outward-normal estimate with only **four** samples, taken at the vertices of
//! a regular tetrahedron centered on the query point. Each sample is weighted
//! by its offset direction and the four weighted samples are summed; the result
//! is proportional to the gradient and only needs normalization. Trading two
//! samples for a slightly less isotropic estimate is the standard `AAA`
//! micro-optimization when a sphere-traced hit must be shaded cheaply.
//!
//! This module differentiates the same trilinear sampler
//! ([`super::mesh_sdf_raymarch::sample_signed_distance`]) as its six-tap
//! sibling, so the two agree on field evaluation. Everything stays
//! transcendental-free — the only non-linear operation is a single `sqrt` to
//! normalize the summed vector — so the result is reproducible across machines.

use super::mesh_sdf_raymarch::sample_signed_distance;
use super::mesh_signed_distance_field::SignedDistanceField;

/// The four tetrahedron vertex offset directions (unnormalized), reused as the
/// per-sample weights. Their component signs form a regular tetrahedron so the
/// weighted sum of field samples approximates the gradient.
const TETRAHEDRON_OFFSETS: [[f32; 3]; 4] = [
    [1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, 1.0],
];

/// Estimates the (unnormalized) gradient of the signed distance field at
/// `point` with the four-sample tetrahedron technique.
///
/// Each of the four [`TETRAHEDRON_OFFSETS`] directions is scaled by one voxel
/// edge, added to `point`, sampled, and accumulated weighted by that same
/// offset. The summed vector points outward (toward increasing signed
/// distance); its magnitude depends on the step and is not normalized here.
pub fn sdf_tetrahedron_gradient(field: &SignedDistanceField, point: [f32; 3]) -> [f32; 3] {
    let step = field.voxel_size();
    let mut gradient = [0f32; 3];
    for offset in &TETRAHEDRON_OFFSETS {
        let sample_point = [
            point[0] + offset[0] * step,
            point[1] + offset[1] * step,
            point[2] + offset[2] * step,
        ];
        let value = sample_signed_distance(field, sample_point);
        gradient[0] += offset[0] * value;
        gradient[1] += offset[1] * value;
        gradient[2] += offset[2] * value;
    }
    gradient
}

/// Returns the outward unit surface normal at `point` using the tetrahedron
/// technique (the normalized [`sdf_tetrahedron_gradient`]), or `None` when the
/// summed vector is too short to orient reliably (for example in a flat,
/// constant region of the field).
pub fn sdf_tetrahedron_normal(field: &SignedDistanceField, point: [f32; 3]) -> Option<[f32; 3]> {
    let gradient = sdf_tetrahedron_gradient(field, point);
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
    use crate::ray_scene::mesh_sdf_normal::sdf_normal;
    use crate::ray_scene::mesh_signed_distance_field::signed_distance_field;
    use crate::ray_scene::mesh_voxel_padding::pad_voxel_grid;
    use crate::ray_scene::mesh_voxelize::voxelize_surface;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh from positions (no normals/uvs).
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// Closed axis-aligned unit cube (12 triangles) spanning `[0, 1]^3`.
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
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 2, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 7, 4],
            [1, 2, 6],
            [1, 6, 5],
        ];
        mesh(p, i)
    }

    /// Padded signed distance field of the unit cube (resolution 8, margin 4)
    /// so exterior cells carry real positive distances for a clean gradient.
    fn padded_cube_field() -> SignedDistanceField {
        let grid = voxelize_surface(&cube(), 8).unwrap();
        let padded = pad_voxel_grid(&grid, 4);
        signed_distance_field(&padded)
    }

    /// Just outside the +z (top) face of the cube, centered in x and y.
    fn above_top_face(field: &SignedDistanceField) -> [f32; 3] {
        [0.5, 0.5, 1.0 + field.voxel_size()]
    }

    #[test]
    fn normal_points_outward_near_top_face() {
        let field = padded_cube_field();
        let normal =
            sdf_tetrahedron_normal(&field, above_top_face(&field)).expect("gradient must orient");
        // The dominant component points along +z, away from the solid.
        assert!(
            normal[2] > 0.7,
            "normal {normal:?} should point along +z above the top face",
        );
    }

    #[test]
    fn normal_is_unit_length() {
        let field = padded_cube_field();
        let normal =
            sdf_tetrahedron_normal(&field, above_top_face(&field)).expect("gradient must orient");
        let length_sq =
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        assert!((length_sq - 1.0).abs() < 1e-5, "length^2 {length_sq} ~ 1");
    }

    #[test]
    fn agrees_with_central_difference_normal() {
        let field = padded_cube_field();
        let point = above_top_face(&field);
        let tetra =
            sdf_tetrahedron_normal(&field, point).expect("tetra normal must orient");
        let central = sdf_normal(&field, point).expect("central normal must orient");
        // Both estimate the same outward normal; cosine close to 1.
        let dot = tetra[0] * central[0] + tetra[1] * central[1] + tetra[2] * central[2];
        assert!(dot > 0.98, "tetra {tetra:?} vs central {central:?}, dot {dot}");
    }

    #[test]
    fn flat_region_has_no_orientation() {
        // A single-cell degenerate field is constant everywhere, so the
        // tetrahedron gradient collapses to zero and no normal is returned.
        let grid = voxelize_surface(&cube(), 1).unwrap();
        let field = signed_distance_field(&grid);
        let center = [
            field.origin()[0] + 0.5 * field.voxel_size(),
            field.origin()[1] + 0.5 * field.voxel_size(),
            field.origin()[2] + 0.5 * field.voxel_size(),
        ];
        assert!(sdf_tetrahedron_normal(&field, center).is_none());
    }
}
