//! Distance-field material thickness probing for the `CPU` golden path.
//!
//! Translucency and subsurface scattering need to know how much solid lies
//! behind a shaded point: a thin ear or leaf lets light bleed through, a thick
//! torso does not. `AAA` renderers bake this as a "thickness map" by shooting a
//! short ray *into* the surface and measuring how far the interior extends. A
//! signed distance field answers the same question directly — the field is
//! negative inside the mesh — so marching inward along the surface normal and
//! finding where the sign flips back to positive yields the local thickness.
//!
//! This module walks from a surface point along the inward normal (the
//! negated outward normal), sampling the trilinear field
//! ([`super::mesh_sdf_raymarch::sample_signed_distance`]). It records when the
//! march enters the solid (sampled distance turns negative) and returns the
//! travelled distance at the first exit back into free space. When the solid
//! is thicker than the probe budget the march returns the full budget, and a
//! point whose inward ray never enters the solid reports zero thickness.
//!
//! The probe uses only multiplies, adds, and comparisons apart from the `sqrt`
//! the normal normalization performs, so it is transcendental-free and
//! reproducible. [`sdf_thickness`] returns a distance in `0..=max_distance`.

use super::mesh_signed_distance_field::SignedDistanceField;
use super::mesh_sdf_raymarch::sample_signed_distance;

/// Probes the solid thickness behind `position` along the inward normal,
/// returning the distance in `0..=max_distance`.
///
/// `normal` is the outward surface normal (normalized internally); the probe
/// marches along its negation into the mesh. It advances in fixed `step`
/// increments for up to `max_steps` iterations or `max_distance` world units,
/// whichever comes first. The returned value is the distance travelled when
/// the march first re-emerges from the solid; if the interior extends past the
/// budget the full `max_distance` is returned, and if the inward ray never
/// enters the solid the result is zero.
///
/// Returns [`None`] when `normal` is too short to normalize.
pub fn sdf_thickness(
    field: &SignedDistanceField,
    position: [f32; 3],
    normal: [f32; 3],
    max_distance: f32,
    step: f32,
    max_steps: u32,
) -> Option<f32> {
    let n = normalize(normal)?;
    // March along the inward normal (negated outward normal).
    let inward = [-n[0], -n[1], -n[2]];

    let mut t = 0.0f32;
    let mut entered_solid = false;
    for _ in 0..max_steps {
        if t > max_distance {
            break;
        }
        let sample = [
            position[0] + t * inward[0],
            position[1] + t * inward[1],
            position[2] + t * inward[2],
        ];
        let distance = sample_signed_distance(field, sample);
        if distance < 0.0 {
            entered_solid = true;
        } else if entered_solid {
            // Re-emerged into free space: `t` is the traversed thickness.
            return Some(t.min(max_distance));
        }
        t += step;
    }

    // The march never exited: the full budget when still inside, else no solid.
    Some(if entered_solid { max_distance } else { 0.0 })
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
    use super::sdf_thickness;
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
            sdf_thickness(&field, [0.5, 0.5, 1.0], [0.0, 0.0, 0.0], 2.0, 0.02, 256).is_none(),
            "a zero-length normal cannot be normalized",
        );
    }

    #[test]
    fn face_center_measures_full_depth() {
        let field = padded_cube_field();
        // Top face centre, outward normal up: inward march spans the unit cube.
        let thickness =
            sdf_thickness(&field, [0.5, 0.5, 1.0], [0.0, 0.0, 1.0], 2.0, 0.01, 400).unwrap();
        assert!(
            (thickness - 1.0).abs() < 0.1,
            "cube depth along the normal is ~1 (got {thickness})",
        );
    }

    #[test]
    fn inward_ray_in_open_space_is_zero() {
        let field = padded_cube_field();
        // Left of the cube, normal pointing toward it: the inward march heads
        // away into empty exterior space and never enters the solid.
        let thickness =
            sdf_thickness(&field, [-0.4, 0.5, 0.5], [1.0, 0.0, 0.0], 2.0, 0.02, 256).unwrap();
        assert_eq!(thickness, 0.0, "no solid lies along the inward ray");
    }

    #[test]
    fn budget_caps_a_thick_solid() {
        let field = padded_cube_field();
        // A probe budget shorter than the cube depth returns the full budget.
        let budget = 0.3;
        let thickness =
            sdf_thickness(&field, [0.5, 0.5, 1.0], [0.0, 0.0, 1.0], budget, 0.01, 400).unwrap();
        assert_eq!(thickness, budget, "the interior extends past the budget");
    }

    #[test]
    fn result_stays_in_range() {
        let field = padded_cube_field();
        let max_distance = 2.0;
        let thickness =
            sdf_thickness(&field, [0.5, 0.5, 1.0], [0.0, 0.0, 1.0], max_distance, 0.01, 400)
                .unwrap();
        assert!(
            (0.0..=max_distance).contains(&thickness),
            "thickness stays within the probe budget",
        );
    }
}
