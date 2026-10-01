//! Distance-field soft shadows via sphere-traced penumbra estimation for the
//! `CPU` golden path.
//!
//! A hard shadow ray only answers "is the light blocked?"; a *soft* shadow
//! also estimates how much of the light's disc is occluded, producing the
//! smooth penumbra that `AAA` renderers rely on for grounded, believable
//! contact shadows. Inigo Quilez's technique computes this for free during a
//! sphere trace: at every march step the ratio of the sampled distance to the
//! travelled distance bounds the half-angle of the cone that reached the
//! surface unobstructed, and the running minimum of that ratio approximates
//! the visible fraction of the light.
//!
//! This module marches from a surface point toward the light through the
//! signed distance field ([`super::mesh_signed_distance_field::SignedDistanceField`])
//! using the trilinear sampler ([`super::mesh_sdf_raymarch::sample_signed_distance`]).
//! It implements IQ's improved estimator, which corrects the naive `k*h/t`
//! ratio for the distance already stepped past the closest approach, removing
//! the banding the simple form shows. Marching outside the mesh requires the
//! exterior shell added by [`super::mesh_voxel_padding::pad_voxel_grid`]; the
//! sampler clamps to border values past the grid, so a ray that leaves the
//! lattice simply stops accumulating occlusion.
//!
//! The estimator uses only multiplies, a `max`-guarded `sqrt`, and a step
//! floor, so it is transcendental-free and reproducible. [`sdf_soft_shadow`]
//! returns the visible fraction in `0..=1` (one fully lit, zero fully
//! shadowed) together with the step count and whether the ray struck the
//! surface.

use super::mesh_signed_distance_field::SignedDistanceField;
use super::mesh_sdf_raymarch::sample_signed_distance;

/// Result of a distance-field soft-shadow trace.
///
/// Returned by [`sdf_soft_shadow`]. `visibility` is the estimated unoccluded
/// fraction of the light in `0..=1`, `steps` reports how many march
/// iterations were taken, and `occluded` is `true` when the ray reached the
/// surface (a hard hit, visibility zero).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftShadow {
    /// Estimated visible fraction of the light in `0..=1`.
    visibility: f32,
    /// Number of march iterations performed.
    steps: u32,
    /// Whether the ray struck the surface (fully shadowed).
    occluded: bool,
}

impl SoftShadow {
    /// Estimated visible fraction of the light in `0..=1` (one fully lit).
    pub fn visibility(&self) -> f32 {
        self.visibility
    }

    /// Number of march iterations performed.
    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// Whether the ray struck the surface (fully shadowed).
    pub fn occluded(&self) -> bool {
        self.occluded
    }
}

/// Traces a soft shadow ray through `field` and returns the estimated visible
/// fraction of the light.
///
/// `origin` is the shaded surface point and `direction` points toward the
/// light; `direction` is normalized internally, so `min_distance` and
/// `max_distance` are world-space march bounds (start `min_distance` slightly
/// off the surface to avoid self-shadowing). `sharpness` is IQ's penumbra
/// factor: larger values narrow the penumbra toward a hard shadow. The march
/// stops early with full occlusion when the sampled distance falls to
/// `hit_epsilon`, and otherwise runs until `max_distance` or `max_steps`.
///
/// Returns [`None`] only when `direction` is too short to normalize; a ray
/// that reaches no occluder is a valid result reporting `visibility == 1`.
pub fn sdf_soft_shadow(
    field: &SignedDistanceField,
    origin: [f32; 3],
    direction: [f32; 3],
    min_distance: f32,
    max_distance: f32,
    sharpness: f32,
    hit_epsilon: f32,
    max_steps: u32,
) -> Option<SoftShadow> {
    let dir = normalize(direction)?;

    // Floor on each step so a near-zero sample cannot freeze the march.
    let min_step = field.voxel_size() * 0.125;

    let mut visibility = 1.0f32;
    let mut t = min_distance.max(0.0);
    // Sentinel so the previous-sample correction `y` is zero on the first step.
    let mut prev_distance = f32::INFINITY;
    let mut steps = 0u32;
    let mut occluded = false;

    for step in 0..max_steps {
        if t > max_distance {
            break;
        }
        steps = step + 1;

        let position = [
            origin[0] + t * dir[0],
            origin[1] + t * dir[1],
            origin[2] + t * dir[2],
        ];
        let distance = sample_signed_distance(field, position);
        if distance <= hit_epsilon {
            visibility = 0.0;
            occluded = true;
            break;
        }

        // IQ's improved penumbra: subtract the overshoot `y` past the closest
        // approach so the cone half-angle is measured from the true nearest
        // point rather than the current sample.
        let y = if prev_distance.is_finite() {
            distance * distance / (2.0 * prev_distance)
        } else {
            0.0
        };
        let chord = (distance * distance - y * y).max(0.0).sqrt();
        let reach = (t - y).max(min_step);
        visibility = visibility.min(sharpness * chord / reach);

        prev_distance = distance;
        t += distance.max(min_step);
    }

    Some(SoftShadow {
        visibility: visibility.clamp(0.0, 1.0),
        steps,
        occluded,
    })
}

/// Returns the unit-length direction, or [`None`] when `direction` is too
/// short to normalize reliably.
fn normalize(direction: [f32; 3]) -> Option<[f32; 3]> {
    let length_squared =
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    if length_squared <= f32::MIN_POSITIVE {
        return None;
    }
    let length = length_squared.sqrt();
    Some([
        direction[0] / length,
        direction[1] / length,
        direction[2] / length,
    ])
}

#[cfg(test)]
mod tests {
    use super::sdf_soft_shadow;
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
    fn degenerate_direction_returns_none() {
        let field = padded_cube_field();
        assert!(
            sdf_soft_shadow(&field, [0.5, 0.5, -0.3], [0.0, 0.0, 0.0], 0.05, 2.0, 8.0, 0.01, 128)
                .is_none(),
            "a zero-length direction cannot be normalized",
        );
    }

    #[test]
    fn ray_through_the_cube_is_fully_occluded() {
        let field = padded_cube_field();
        // Start just below the cube, aim straight up through its centre.
        let result = sdf_soft_shadow(
            &field,
            [0.5, 0.5, -0.3],
            [0.0, 0.0, 1.0],
            0.05,
            2.0,
            8.0,
            0.01,
            256,
        )
        .unwrap();
        assert!(result.occluded(), "the ray crosses the cube");
        assert_eq!(result.visibility(), 0.0, "a hit is fully shadowed");
    }

    #[test]
    fn ray_clear_of_the_cube_is_fully_lit() {
        let field = padded_cube_field();
        // Column well outside the cube's cross-section: never approaches it.
        let result = sdf_soft_shadow(
            &field,
            [-0.4, -0.4, -0.3],
            [0.0, 0.0, 1.0],
            0.05,
            2.0,
            8.0,
            0.01,
            256,
        )
        .unwrap();
        assert!(!result.occluded(), "the ray misses the cube");
        assert_eq!(result.visibility(), 1.0, "an unobstructed ray is fully lit");
    }

    #[test]
    fn grazing_ray_is_in_partial_penumbra() {
        let field = padded_cube_field();
        // Just outside the x face: approaches but never touches the surface.
        let result = sdf_soft_shadow(
            &field,
            [-0.1, 0.5, -0.3],
            [0.0, 0.0, 1.0],
            0.05,
            2.0,
            8.0,
            0.01,
            256,
        )
        .unwrap();
        assert!(!result.occluded(), "the grazing ray does not hit");
        let v = result.visibility();
        assert!((0.0..=1.0).contains(&v), "visibility stays in range");
        assert!(v < 1.0, "the near approach darkens the penumbra");
        assert!(v > 0.0, "a miss is never fully shadowed");
    }

    #[test]
    fn higher_sharpness_lightens_the_penumbra() {
        let field = padded_cube_field();
        let soft = sdf_soft_shadow(
            &field,
            [-0.1, 0.5, -0.3],
            [0.0, 0.0, 1.0],
            0.05,
            2.0,
            4.0,
            0.01,
            256,
        )
        .unwrap();
        let sharp = sdf_soft_shadow(
            &field,
            [-0.1, 0.5, -0.3],
            [0.0, 0.0, 1.0],
            0.05,
            2.0,
            16.0,
            0.01,
            256,
        )
        .unwrap();
        assert!(
            sharp.visibility() >= soft.visibility(),
            "a sharper penumbra is at least as bright",
        );
    }

    #[test]
    fn zero_steps_leaves_the_light_unoccluded() {
        let field = padded_cube_field();
        let result = sdf_soft_shadow(
            &field,
            [0.5, 0.5, -0.3],
            [0.0, 0.0, 1.0],
            0.05,
            2.0,
            8.0,
            0.01,
            0,
        )
        .unwrap();
        assert_eq!(result.steps(), 0, "no march iterations ran");
        assert_eq!(result.visibility(), 1.0, "no occlusion was accumulated");
    }
}
