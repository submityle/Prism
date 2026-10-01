//! Distance-field cone-traced occlusion with bent normals for the `CPU`
//! golden path.
//!
//! The five-tap estimator in [`super::mesh_sdf_ambient_occlusion`] returns a
//! single scalar visibility along the surface normal. Modern `AAA` renderers
//! want more: Unreal's Distance Field Ambient Occlusion sweeps a small fan of
//! *cones* across the hemisphere and keeps, per cone, the fraction of the cone
//! cross-section the field leaves unblocked. Averaging the cone visibilities
//! gives a far less noisy occlusion term, and the visibility-weighted average
//! of the cone directions gives the **bent normal** — the mean unoccluded
//! direction — which image-based lighting and global illumination sample
//! instead of the geometric normal so shading leans away from nearby
//! occluders.
//!
//! Each cone is marched like a sphere trace, but instead of stopping at the
//! surface the march keeps the running minimum of `sampled_distance /
//! cone_radius`: where the field stays wider than the cone the sample is open
//! (`1`), where it pinches below the cone radius the cone is partially blocked,
//! and a negative (interior) sample fully occludes it (`0`). The hemisphere
//! fan is a baked center-plus-six-ring pattern rotated into the surface frame
//! by the branchless orthonormal basis of Duff et al. (2017), so no
//! trigonometry runs at evaluation time.
//!
//! Everything is transcendental-free: only `abs`, `min`, `max`, `clamp`,
//! `signum`, baked ring constants, and the `sqrt` inside vector normalization
//! are used, so the estimate is reproducible across machines.

use super::mesh_sdf_raymarch::sample_signed_distance;
use super::mesh_signed_distance_field::SignedDistanceField;

/// Cone-traced occlusion result at a shaded point: a scalar visibility factor
/// and the bent normal (mean unoccluded direction).
///
/// Returned by [`sdf_cone_occlusion`]. `visibility` is in `0..=1` (one fully
/// open, zero fully occluded); `bent_normal` is a unit vector pointing toward
/// the least-occluded part of the hemisphere, falling back to the geometric
/// normal when every cone is equally open.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConeOcclusion {
    /// Hemisphere visibility factor in `0..=1` (one fully open).
    visibility: f32,
    /// Unit bent normal: the visibility-weighted mean cone direction.
    bent_normal: [f32; 3],
}

impl ConeOcclusion {
    /// Hemisphere visibility factor in `0..=1` (one fully open).
    pub fn visibility(&self) -> f32 {
        self.visibility
    }

    /// Unit bent normal (mean unoccluded direction).
    pub fn bent_normal(&self) -> [f32; 3] {
        self.bent_normal
    }
}

/// One over the square root of two (`cos 45° = sin 45°`), the ring elevation.
const COS45: f32 = 0.707_106_77;
/// Half of [`COS45`] (`sin 45° · cos 60°`), a ring tangent `x`/`y` magnitude.
const RING_HALF: f32 = 0.353_553_38;
/// `sin 45° · sin 60°` (`= √6 / 4`), the other ring tangent magnitude.
const RING_TALL: f32 = 0.612_372_44;

/// The seven cone directions in tangent space (`z` is the surface normal): the
/// center cone along the normal plus a six-way ring at 45° elevation. Baked so
/// no trigonometry runs at evaluation time; each cone's weight is its `z`
/// (cosine of the elevation from the normal).
const CONE_DIRS: [[f32; 3]; 7] = [
    [0.0, 0.0, 1.0],
    [COS45, 0.0, COS45],
    [RING_HALF, RING_TALL, COS45],
    [-RING_HALF, RING_TALL, COS45],
    [-COS45, 0.0, COS45],
    [-RING_HALF, -RING_TALL, COS45],
    [RING_HALF, -RING_TALL, COS45],
];

/// Cone-traces hemisphere occlusion at `position` with surface `normal`,
/// returning the visibility factor and bent normal.
///
/// Seven cones ([`CONE_DIRS`]) are rotated into the surface frame and marched
/// out to `max_distance` in at most `step_count` steps each. A cone's
/// visibility is the running minimum of `sampled_distance / cone_radius`, where
/// the cone radius grows as `tan_half_angle · t`; a wider-than-cone field keeps
/// it open, a pinch narrows it, and an interior sample closes it. The returned
/// `visibility` is the cosine-weighted mean cone visibility and `bent_normal`
/// is the cosine- and visibility-weighted mean cone direction, renormalized.
///
/// `normal` is normalized internally and should point away from the surface.
/// `tan_half_angle` is the tangent of each cone's half-angle (wider cones catch
/// more occluders; `0.5` is a reasonable default). Returns [`None`] when
/// `normal` is too short to normalize.
pub fn sdf_cone_occlusion(
    field: &SignedDistanceField,
    position: [f32; 3],
    normal: [f32; 3],
    tan_half_angle: f32,
    max_distance: f32,
    step_count: u32,
) -> Option<ConeOcclusion> {
    let n = normalize(normal)?;
    let (tangent, bitangent) = orthonormal_basis(n);

    // Start just off the surface and never step shorter than this so a grazing
    // sample cannot freeze the march.
    let min_step = field.voxel_size() * 0.5;

    let mut visibility_sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    let mut bent = [0.0f32; 3];

    for cone in &CONE_DIRS {
        // Rotate the tangent-space cone direction into world space.
        let dir = [
            tangent[0] * cone[0] + bitangent[0] * cone[1] + n[0] * cone[2],
            tangent[1] * cone[0] + bitangent[1] * cone[1] + n[1] * cone[2],
            tangent[2] * cone[0] + bitangent[2] * cone[1] + n[2] * cone[2],
        ];
        // Weight by the cone's elevation cosine (its tangent-space z).
        let weight = cone[2];

        let mut cone_visibility = 1.0f32;
        let mut t = min_step;
        for _ in 0..step_count {
            if t >= max_distance {
                break;
            }
            let sample_point = [
                position[0] + t * dir[0],
                position[1] + t * dir[1],
                position[2] + t * dir[2],
            ];
            let distance = sample_signed_distance(field, sample_point);
            let radius = tan_half_angle * t;
            // Fraction of the cone cross-section left open at this step.
            let open = if radius <= f32::MIN_POSITIVE {
                1.0
            } else {
                (distance / radius).clamp(0.0, 1.0)
            };
            cone_visibility = cone_visibility.min(open);
            if cone_visibility <= 0.0 {
                break;
            }
            t += distance.max(min_step);
        }

        visibility_sum += cone_visibility * weight;
        weight_sum += weight;
        bent[0] += dir[0] * cone_visibility * weight;
        bent[1] += dir[1] * cone_visibility * weight;
        bent[2] += dir[2] * cone_visibility * weight;
    }

    let visibility = if weight_sum > f32::MIN_POSITIVE {
        (visibility_sum / weight_sum).clamp(0.0, 1.0)
    } else {
        1.0
    };
    // Renormalize the accumulated direction; fall back to the geometric normal
    // when every cone is fully occluded (zero-length sum).
    let bent_normal = normalize(bent).unwrap_or(n);

    Some(ConeOcclusion {
        visibility,
        bent_normal,
    })
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

/// Builds a right-handed orthonormal basis `(tangent, bitangent)` perpendicular
/// to the unit `normal` with the branchless method of Duff et al. (2017).
///
/// Uses only `signum`, multiplies, and adds — no trigonometry — and stays
/// numerically stable across the whole sphere because the sign of the normal's
/// `z` component is folded in to avoid the degenerate pole.
fn orthonormal_basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let sign = normal[2].signum();
    let a = -1.0 / (sign + normal[2]);
    let b = normal[0] * normal[1] * a;
    let tangent = [
        1.0 + sign * normal[0] * normal[0] * a,
        sign * b,
        -sign * normal[0],
    ];
    let bitangent = [b, sign + normal[1] * normal[1] * a, -normal[1]];
    (tangent, bitangent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::mesh_signed_distance_field::signed_distance_field;
    use crate::ray_scene::mesh_voxel_padding::pad_voxel_grid;
    use crate::ray_scene::mesh_voxelize::voxelize_surface;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh (no normals/uvs) for test fixtures.
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

    /// Padded signed distance field of the unit cube with an exterior shell.
    fn padded_cube_field() -> SignedDistanceField {
        let grid = voxelize_surface(&cube(), 8).unwrap();
        let padded = pad_voxel_grid(&grid, 4);
        signed_distance_field(&padded)
    }

    /// Axis-aligned box mesh spanning `[min, max]` (12 triangles), with its
    /// index block offset by `base` so several boxes can share one mesh.
    fn box_mesh(min: [f32; 3], max: [f32; 3], base: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let p = vec![
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
        ];
        let tris = [
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
        let i = tris
            .iter()
            .map(|t| [t[0] + base, t[1] + base, t[2] + base])
            .collect();
        (p, i)
    }

    /// Concave fixture: a unit-cube floor at `[0, 1]^3` with a taller wall block
    /// rising on its `+x` side (`x in [1, 1.5]`, `z in [0, 2]`). Standing on the
    /// floor near the `+x` edge, cones toward `+x` are blocked by the wall while
    /// `-x` stays open, so the bent normal must lean toward `-x`.
    fn floor_and_wall_field() -> SignedDistanceField {
        let (mut positions, mut indices) = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        let (wall_p, wall_i) = box_mesh([1.0, 0.0, 0.0], [1.5, 1.0, 2.0], positions.len() as u32);
        positions.extend(wall_p);
        indices.extend(wall_i);
        let combined = mesh(positions, indices);
        let grid = voxelize_surface(&combined, 10).unwrap();
        let padded = pad_voxel_grid(&grid, 4);
        signed_distance_field(&padded)
    }

    #[test]
    fn orthonormal_basis_is_orthonormal() {
        for n in [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            normalize([0.3, -0.7, 0.65]).unwrap(),
        ] {
            let (t, b) = orthonormal_basis(n);
            let dot = |a: [f32; 3], c: [f32; 3]| a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
            assert!((dot(t, t) - 1.0).abs() < 1e-4, "tangent unit length");
            assert!((dot(b, b) - 1.0).abs() < 1e-4, "bitangent unit length");
            assert!(dot(t, n).abs() < 1e-4, "tangent perpendicular to normal");
            assert!(dot(b, n).abs() < 1e-4, "bitangent perpendicular to normal");
            assert!(dot(t, b).abs() < 1e-4, "tangent perpendicular to bitangent");
        }
    }

    #[test]
    fn degenerate_normal_returns_none() {
        let field = padded_cube_field();
        assert!(
            sdf_cone_occlusion(&field, [0.5, 0.5, 1.3], [0.0, 0.0, 0.0], 0.5, 1.0, 32).is_none(),
            "a zero-length normal cannot be normalized",
        );
    }

    #[test]
    fn open_space_is_fully_visible_with_normal_bent() {
        let field = padded_cube_field();
        // Far above the cube: the whole hemisphere is empty.
        let r = sdf_cone_occlusion(&field, [0.5, 0.5, 2.0], [0.0, 0.0, 1.0], 0.5, 1.0, 32).unwrap();
        assert!(r.visibility() > 0.95, "open hemisphere (vis = {})", r.visibility());
        // With nothing to deflect it, the bent normal stays near the normal.
        assert!(r.bent_normal()[2] > 0.95, "bent normal ~ +z: {:?}", r.bent_normal());
    }

    #[test]
    fn facing_into_solid_is_more_occluded_than_facing_away() {
        let field = padded_cube_field();
        let away =
            sdf_cone_occlusion(&field, [0.5, 0.5, 1.05], [0.0, 0.0, 1.0], 0.5, 1.0, 48).unwrap();
        let into =
            sdf_cone_occlusion(&field, [0.5, 0.5, 1.05], [0.0, 0.0, -1.0], 0.5, 1.0, 48).unwrap();
        assert!(
            away.visibility() > into.visibility(),
            "away {} should beat into {}",
            away.visibility(),
            into.visibility(),
        );
    }

    #[test]
    fn bent_normal_is_unit_length() {
        let field = padded_cube_field();
        let r =
            sdf_cone_occlusion(&field, [0.5, 0.5, 1.05], [0.0, 0.0, 1.0], 0.5, 1.0, 48).unwrap();
        let bn = r.bent_normal();
        let len_sq = bn[0] * bn[0] + bn[1] * bn[1] + bn[2] * bn[2];
        assert!((len_sq - 1.0).abs() < 1e-4, "bent normal unit length ({len_sq})");
        assert!((0.0..=1.0).contains(&r.visibility()), "visibility in range");
    }

    #[test]
    fn bent_normal_deflects_away_from_a_nearby_wall() {
        let field = floor_and_wall_field();
        // On the floor just below the top, near the +x wall. Cones toward +x are
        // blocked by the wall; -x is open, so the bent normal leans toward -x.
        let r =
            sdf_cone_occlusion(&field, [0.9, 0.5, 0.95], [0.0, 0.0, 1.0], 0.6, 1.5, 64).unwrap();
        assert!(
            r.bent_normal()[0] < -0.02,
            "bent normal should lean away from the +x wall: {:?}",
            r.bent_normal(),
        );
        // The occluded hemisphere is also measurably darker than fully open.
        assert!(r.visibility() < 0.99, "wall occludes part of the cone fan");
    }
}
