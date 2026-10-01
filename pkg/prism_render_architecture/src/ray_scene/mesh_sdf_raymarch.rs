//! Sphere-traced ray marching of a signed distance field for the `CPU` golden
//! path.
//!
//! Sphere tracing is the canonical way `AAA` renderers consume a signed
//! distance field (`SDF`): because the field value at any point is a lower
//! bound on the distance to the nearest surface, a ray can safely advance by
//! exactly that value every step and converge on the first intersection
//! without ever overshooting. The same traversal drives distance-field soft
//! shadows, ambient occlusion, and global-illumination cone stepping.
//!
//! This module is the direct consumer of
//! [`super::mesh_signed_distance_field::SignedDistanceField`]. It adds two
//! operations: continuous trilinear sampling of the discrete field
//! ([`sample_signed_distance`]) and a sphere tracer that marches a ray through
//! the grid's world-space bounding box ([`sphere_trace`]). Everything is
//! transcendental-free — only `floor`, `abs`, `min`, `max`, `clamp`, linear
//! interpolation, and a single `sqrt` for direction normalization are used —
//! so the traversal stays reproducible across machines.

use super::mesh_signed_distance_field::SignedDistanceField;

/// A sphere-tracing hit along a ray through a signed distance field.
///
/// Returned by [`sphere_trace`] when the march reaches a point whose sampled
/// signed distance drops to or below the caller's hit epsilon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfHit {
    /// Ray parameter of the hit measured along the (normalized) direction, so
    /// it equals the world-space distance travelled from the ray origin.
    t: f32,
    /// World-space position of the hit (`origin + t * direction`).
    position: [f32; 3],
    /// Sampled signed distance at the hit position (at or below the epsilon).
    distance: f32,
    /// Number of marching steps taken to reach the hit.
    steps: u32,
}

impl SdfHit {
    /// Ray parameter of the hit, equal to the world-space distance travelled.
    pub fn t(&self) -> f32 {
        self.t
    }

    /// World-space position of the hit.
    pub fn position(&self) -> [f32; 3] {
        self.position
    }

    /// Sampled signed distance at the hit position.
    pub fn distance(&self) -> f32 {
        self.distance
    }

    /// Number of marching steps taken to reach the hit.
    pub fn steps(&self) -> u32 {
        self.steps
    }
}

/// Samples the continuous signed distance at an arbitrary world-space `point`
/// by trilinearly interpolating the eight surrounding cell-center values.
///
/// The field stores one signed distance per cell center; this maps `point`
/// into continuous cell-center space, locates the lower-corner cell, and
/// blends its eight neighbors. Points outside the grid clamp to the border
/// cells (nearest-value extrapolation). A degenerate axis of a single cell
/// contributes no interpolation weight. When `point` lands exactly on a cell
/// center the result equals that cell's stored
/// [`SignedDistanceField::signed_distance`].
pub fn sample_signed_distance(field: &SignedDistanceField, point: [f32; 3]) -> f32 {
    let dims = field.dims();
    let origin = field.origin();
    let voxel_size = field.voxel_size();

    // Continuous cell-center coordinate on each axis, clamped into the valid
    // cell range so points outside the grid use border values.
    let mut base = [0u32; 3];
    let mut frac = [0f32; 3];
    for (axis, slot) in base.iter_mut().enumerate() {
        let last = dims[axis].saturating_sub(1);
        if dims[axis] <= 1 {
            // Degenerate axis: single layer, no interpolation.
            *slot = 0;
            frac[axis] = 0.0;
            continue;
        }
        let continuous = (point[axis] - origin[axis]) / voxel_size - 0.5;
        let clamped = continuous.clamp(0.0, last as f32);
        // Lower corner cannot exceed the second-to-last cell so `base + 1`
        // always stays inside the grid.
        let lower = clamped.floor().clamp(0.0, (last - 1) as f32);
        *slot = lower as u32;
        frac[axis] = clamped - lower;
    }

    // Fetch the eight surrounding cell-center values.
    let corner = |dx: u32, dy: u32, dz: u32| -> f32 {
        let coord = [
            (base[0] + dx).min(dims[0] - 1),
            (base[1] + dy).min(dims[1] - 1),
            (base[2] + dz).min(dims[2] - 1),
        ];
        field.signed_distance(coord)
    };

    let d000 = corner(0, 0, 0);
    let d100 = corner(1, 0, 0);
    let d010 = corner(0, 1, 0);
    let d110 = corner(1, 1, 0);
    let d001 = corner(0, 0, 1);
    let d101 = corner(1, 0, 1);
    let d011 = corner(0, 1, 1);
    let d111 = corner(1, 1, 1);

    let c00 = lerp(d000, d100, frac[0]);
    let c01 = lerp(d001, d101, frac[0]);
    let c10 = lerp(d010, d110, frac[0]);
    let c11 = lerp(d011, d111, frac[0]);
    let c0 = lerp(c00, c10, frac[1]);
    let c1 = lerp(c01, c11, frac[1]);
    lerp(c0, c1, frac[2])
}

/// Sphere-traces a ray against a signed distance field, returning the first
/// surface intersection (where the sampled distance falls to `hit_epsilon`).
///
/// The ray is first clipped to the grid's world-space bounding box; marching
/// starts at the box entry and never samples outside the field. `direction`
/// is normalized internally, so `t` and `max_distance` are measured in world
/// units. Each step advances by the sampled distance (floored by a small
/// fraction of a voxel to avoid stalling on near-surface plateaus). Returns
/// `None` when the ray misses the box, exits the box, exceeds `max_distance`,
/// or runs out of `max_steps` without converging.
pub fn sphere_trace(
    field: &SignedDistanceField,
    origin: [f32; 3],
    direction: [f32; 3],
    max_distance: f32,
    hit_epsilon: f32,
    max_steps: u32,
) -> Option<SdfHit> {
    let dir = normalize(direction)?;

    let dims = field.dims();
    let grid_origin = field.origin();
    let voxel_size = field.voxel_size();
    let grid_max = [
        grid_origin[0] + dims[0] as f32 * voxel_size,
        grid_origin[1] + dims[1] as f32 * voxel_size,
        grid_origin[2] + dims[2] as f32 * voxel_size,
    ];

    let (t_enter, t_exit) = ray_aabb(origin, dir, grid_origin, grid_max)?;
    let t_exit = t_exit.min(max_distance);
    if t_enter > t_exit {
        return None;
    }

    // Floor on each step so a near-zero sample cannot freeze the march.
    let min_step = voxel_size * 0.125;
    let mut t = t_enter;
    for step in 0..max_steps {
        let position = [
            origin[0] + t * dir[0],
            origin[1] + t * dir[1],
            origin[2] + t * dir[2],
        ];
        let distance = sample_signed_distance(field, position);
        if distance <= hit_epsilon {
            return Some(SdfHit {
                t,
                position,
                distance,
                steps: step + 1,
            });
        }
        t += distance.max(min_step);
        if t > t_exit {
            return None;
        }
    }
    None
}

/// Linear interpolation between `a` and `b` by `s` (no clamping of `s`).
fn lerp(a: f32, b: f32, s: f32) -> f32 {
    a + (b - a) * s
}

/// Returns the unit-length direction, or `None` when `direction` is too short
/// to normalize reliably.
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

/// Intersects a ray with an axis-aligned bounding box using the slab method,
/// returning the clamped entry/exit parameters `(t_enter, t_exit)` with
/// `t_enter >= 0`, or `None` when the ray never overlaps the box.
fn ray_aabb(
    origin: [f32; 3],
    direction: [f32; 3],
    box_min: [f32; 3],
    box_max: [f32; 3],
) -> Option<(f32, f32)> {
    let mut t0 = 0.0f32;
    let mut t1 = f32::INFINITY;
    for axis in 0..3 {
        if direction[axis].abs() <= f32::MIN_POSITIVE {
            // Ray is parallel to this slab: miss unless the origin is between
            // the planes.
            if origin[axis] < box_min[axis] || origin[axis] > box_max[axis] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / direction[axis];
        let mut ta = (box_min[axis] - origin[axis]) * inv;
        let mut tb = (box_max[axis] - origin[axis]) * inv;
        if ta > tb {
            core::mem::swap(&mut ta, &mut tb);
        }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
        if t0 > t1 {
            return None;
        }
    }
    Some((t0, t1))
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

    /// Builds the signed distance field of the unit cube at resolution 4, which
    /// has a verified interior cell at `[1, 1, 1]`.
    fn cube_field() -> SignedDistanceField {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        signed_distance_field(&grid)
    }

    #[test]
    fn ray_from_outside_hits_near_face() {
        let field = cube_field();
        // Aim from well in front of the -z face toward the cube center.
        let origin = [0.5, 0.5, -2.0];
        let hit = sphere_trace(&field, origin, [0.0, 0.0, 1.0], 10.0, 1e-3, 128)
            .expect("ray toward cube center must hit");
        // The entry face sits near z = 0 (grid origin side); allow for the
        // half-voxel sampling offset.
        assert!(hit.position()[2] < 0.5, "hit should be on the near side");
        assert!(hit.t() > 0.0);
        assert!(hit.steps() >= 1);
    }

    #[test]
    fn parallel_offset_ray_misses_box() {
        let field = cube_field();
        // Parallel to +z but well outside the grid in x: never enters the box.
        let hit = sphere_trace(&field, [5.0, 0.5, -2.0], [0.0, 0.0, 1.0], 10.0, 1e-3, 128);
        assert!(hit.is_none(), "offset parallel ray must miss");
    }

    #[test]
    fn backward_ray_misses() {
        let field = cube_field();
        // Pointing away from the cube: the box lies entirely behind the origin.
        let hit = sphere_trace(&field, [0.5, 0.5, -2.0], [0.0, 0.0, -1.0], 10.0, 1e-3, 128);
        assert!(hit.is_none(), "ray pointing away must miss");
    }

    #[test]
    fn interior_sample_is_negative_border_positive() {
        let field = cube_field();
        let voxel = field.voxel_size();
        let origin = field.origin();
        // Center of the cube in world space.
        let center = [
            origin[0] + 2.0 * voxel,
            origin[1] + 2.0 * voxel,
            origin[2] + 2.0 * voxel,
        ];
        assert!(
            sample_signed_distance(&field, center) < 0.0,
            "cube center must sample inside",
        );
        // A solid cube fills its bounding-box grid, so the border cells are
        // surface cells; a point far outside clamps to that shell and samples
        // non-negative (never inside).
        let outside = [origin[0] - 5.0, origin[1] - 5.0, origin[2] - 5.0];
        assert!(
            sample_signed_distance(&field, outside) >= 0.0,
            "point outside the solid must not sample inside",
        );
    }

    #[test]
    fn cell_center_sample_matches_stored_value() {
        let field = cube_field();
        let dims = field.dims();
        let voxel = field.voxel_size();
        let origin = field.origin();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let coord = [x, y, z];
                    // World-space center of this cell.
                    let point = [
                        origin[0] + (x as f32 + 0.5) * voxel,
                        origin[1] + (y as f32 + 0.5) * voxel,
                        origin[2] + (z as f32 + 0.5) * voxel,
                    ];
                    let sampled = sample_signed_distance(&field, point);
                    let stored = field.signed_distance(coord);
                    assert!(
                        (sampled - stored).abs() <= 1e-4,
                        "cell {coord:?}: sampled {sampled} vs stored {stored}",
                    );
                }
            }
        }
    }

    #[test]
    fn non_unit_direction_matches_unit_direction() {
        let field = cube_field();
        let origin = [0.5, 0.5, -2.0];
        let unit = sphere_trace(&field, origin, [0.0, 0.0, 1.0], 10.0, 1e-3, 128)
            .expect("unit direction must hit");
        let scaled = sphere_trace(&field, origin, [0.0, 0.0, 7.5], 10.0, 1e-3, 128)
            .expect("scaled direction must hit");
        // Internal normalization means both report the same hit geometry.
        assert!((unit.t() - scaled.t()).abs() <= 1e-4);
        for axis in 0..3 {
            assert!((unit.position()[axis] - scaled.position()[axis]).abs() <= 1e-4);
        }
    }
}
