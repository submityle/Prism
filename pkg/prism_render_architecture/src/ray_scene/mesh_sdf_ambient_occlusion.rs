//! Distance-field ambient occlusion by normal-cone sampling for the `CPU`
//! golden path.
//!
//! Ambient occlusion darkens creases, contacts, and cavities where nearby
//! geometry blocks the ambient hemisphere, and a signed distance field makes
//! it almost free to estimate: step a few samples out along the surface
//! normal and compare the distance actually travelled before the field
//! predicts a surface against the distance stepped. Where the two agree the
//! point sits in open space; where the field reports a much nearer surface the
//! hemisphere is occluded. This is Inigo Quilez's five-tap estimator, the one
//! `AAA` renderers fall back on for distance-field AO when screen-space data
//! is missing.
//!
//! This module samples the trilinear field
//! ([`super::mesh_sdf_raymarch::sample_signed_distance`]) at increasing offsets
//! along the normal, accumulating the shortfall `(h - d)` weighted by a
//! geometric decay so distant samples matter less, then maps the total to a
//! visibility factor. Sampling away from the surface needs the exterior shell
//! produced by [`super::mesh_voxel_padding::pad_voxel_grid`]; the sampler
//! clamps to border values past the grid, so samples that leave the lattice
//! contribute open space.
//!
//! The estimator uses only multiplies, adds, and a `clamp` apart from the
//! `sqrt` the normal normalization performs, so it is transcendental-free and
//! reproducible. [`sdf_ambient_occlusion`] returns a visibility factor in
//! `0..=1` (one fully open, zero fully occluded).

use super::mesh_signed_distance_field::SignedDistanceField;
use super::mesh_sdf_raymarch::sample_signed_distance;

/// Estimates ambient occlusion at `position` with surface `normal`, returning
/// a visibility factor in `0..=1` (one fully open, zero fully occluded).
///
/// `normal` is normalized internally and should point away from the surface.
/// The estimator takes `sample_count` taps spaced `step` apart along the
/// normal; at each tap it compares the stepped distance against the field's
/// predicted distance and accumulates the shortfall, weighting successive taps
/// by `decay` (a factor in `0..1`, typically `0.95`). `strength` scales the
/// accumulated occlusion before it is mapped to visibility (larger darkens
/// faster, typically `3`).
///
/// Returns [`None`] when `normal` is too short to normalize; a point in fully
/// open space yields `Some(1.0)`.
pub fn sdf_ambient_occlusion(
    field: &SignedDistanceField,
    position: [f32; 3],
    normal: [f32; 3],
    sample_count: u32,
    step: f32,
    decay: f32,
    strength: f32,
) -> Option<f32> {
    let n = normalize(normal)?;

    let mut occlusion = 0.0f32;
    let mut weight = 1.0f32;
    for i in 0..sample_count {
        // Offset grows with the sample index so the taps spread out along the
        // normal; the first tap sits one `step` off the surface.
        let h = step * (i as f32 + 1.0);
        let sample = [
            position[0] + h * n[0],
            position[1] + h * n[1],
            position[2] + h * n[2],
        ];
        let d = sample_signed_distance(field, sample);
        occlusion += (h - d) * weight;
        weight *= decay;
    }

    Some((1.0 - strength * occlusion).clamp(0.0, 1.0))
}

/// Returns the unit-length vector, or [`None`] when `vector` is too short to
/// normalize reliably.
fn normalize(vector: [f32; 3]) -> Option<[f32; 3]> {
    let length_squared = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    if length_squared <= f32::MIN_POSITIVE {
        return None;
    }
    let length = length_squared.sqrt();
    Some([vector[0] / length, vector[1] / length, vector[2] / length])
}

#[cfg(test)]
mod tests {
    use super::sdf_ambient_occlusion;
    use crate::ray_scene::mesh_signed_distance_field::{signed_distance_field, SignedDistanceField};
    use crate::ray_scene::mesh_voxel_padding::pad_voxel_grid;
    use crate::ray_scene::mesh_voxelize::voxelize_surface;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds a triangle mesh with no normals or UVs for test fixtures.
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// Unit cube (12 triangles) spanning `[0, 1]^3`.
    fn cube() -> TriangleMesh {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let indices = vec![
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
        mesh(positions, indices)
    }

    /// Padded signed distance field of the unit cube with an exterior shell.
    fn padded_cube_field() -> SignedDistanceField {
        let grid = voxelize_surface(&cube(), 8).unwrap();
        let padded = pad_voxel_grid(&grid, 4);
        signed_distance_field(&padded)
    }

    #[test]
    fn degenerate_normal_returns_none() {
        let field = padded_cube_field();
        assert!(
            sdf_ambient_occlusion(&field, [0.5, 0.5, 1.05], [0.0, 0.0, 0.0], 5, 0.05, 0.95, 3.0)
                .is_none(),
            "a zero-length normal cannot be normalized",
        );
    }

    #[test]
    fn open_space_is_fully_visible() {
        let field = padded_cube_field();
        // High above the cube, sampling further up into empty space.
        let ao =
            sdf_ambient_occlusion(&field, [0.5, 0.5, 1.3], [0.0, 0.0, 1.0], 5, 0.05, 0.95, 3.0)
                .unwrap();
        assert!(ao > 0.99, "open hemisphere is unoccluded (ao = {ao})");
    }

    #[test]
    fn geometry_in_the_cone_darkens_occlusion() {
        let field = padded_cube_field();
        // Just above the top face, sampling back down toward the solid cube.
        let ao =
            sdf_ambient_occlusion(&field, [0.5, 0.5, 1.05], [0.0, 0.0, -1.0], 5, 0.05, 0.95, 3.0)
                .unwrap();
        assert!(ao < 1.0, "nearby geometry occludes the cone (ao = {ao})");
        assert!((0.0..=1.0).contains(&ao), "visibility stays in range");
    }

    #[test]
    fn facing_away_is_brighter_than_facing_into_geometry() {
        let field = padded_cube_field();
        let away =
            sdf_ambient_occlusion(&field, [0.5, 0.5, 1.05], [0.0, 0.0, 1.0], 5, 0.05, 0.95, 3.0)
                .unwrap();
        let into =
            sdf_ambient_occlusion(&field, [0.5, 0.5, 1.05], [0.0, 0.0, -1.0], 5, 0.05, 0.95, 3.0)
                .unwrap();
        assert!(
            away > into,
            "sampling away from the surface is less occluded ({away} vs {into})",
        );
    }

    #[test]
    fn stronger_strength_darkens_occlusion() {
        let field = padded_cube_field();
        let soft =
            sdf_ambient_occlusion(&field, [0.5, 0.5, 1.05], [0.0, 0.0, -1.0], 5, 0.05, 0.95, 2.0)
                .unwrap();
        let hard =
            sdf_ambient_occlusion(&field, [0.5, 0.5, 1.05], [0.0, 0.0, -1.0], 5, 0.05, 0.95, 6.0)
                .unwrap();
        assert!(
            hard <= soft,
            "a larger strength is at least as dark ({hard} vs {soft})",
        );
    }

    #[test]
    fn zero_samples_is_fully_visible() {
        let field = padded_cube_field();
        let ao =
            sdf_ambient_occlusion(&field, [0.5, 0.5, 1.05], [0.0, 0.0, -1.0], 0, 0.05, 0.95, 3.0)
                .unwrap();
        assert_eq!(ao, 1.0, "no taps means no accumulated occlusion");
    }
}
