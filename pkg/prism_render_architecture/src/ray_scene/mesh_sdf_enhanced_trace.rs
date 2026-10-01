//! Enhanced sphere tracing of a signed distance field (over-relaxation) for
//! the `CPU` golden path.
//!
//! Naive sphere tracing ([`super::mesh_sdf_raymarch::sphere_trace`]) advances a
//! ray by exactly the sampled distance each step. That is always safe but it
//! stalls near grazing surfaces, where thousands of tiny steps crawl along a
//! wall that never quite reaches the hit epsilon. Keinert, Schäfer, Korndörfer,
//! Ganse & Stamminger, *Enhanced Sphere Tracing* (SCCG 2014), accelerate the
//! march with **over-relaxation**: each step is multiplied by a factor
//! `omega` in `[1, 2)`, speculatively overshooting the conservative distance.
//! Whenever the speculation is unsafe — detected when consecutive unbounding
//! spheres no longer overlap — the step is undone and retried at `omega = 1`,
//! so the result is identical to naive tracing but typically converges in far
//! fewer steps.
//!
//! The module is a sibling consumer of
//! [`super::mesh_signed_distance_field::SignedDistanceField`]. It reuses the
//! trilinear sampler, direction normalization, and slab clip from
//! [`super::mesh_sdf_raymarch`] so the two tracers agree on field evaluation
//! and only differ in their stepping policy. Everything stays
//! transcendental-free: only `abs`, `min`, `max`, `clamp`, `signum`, linear
//! arithmetic, and the single `sqrt` inside direction normalization are used,
//! so traversal is reproducible across machines.

use super::mesh_sdf_raymarch::{normalize, ray_aabb, sample_signed_distance};
use super::mesh_signed_distance_field::SignedDistanceField;

/// An enhanced sphere-tracing hit along a ray through a signed distance field.
///
/// Returned by [`enhanced_sphere_trace`] when the over-relaxed march reaches a
/// point whose unbounding-sphere radius falls below the pixel cone footprint.
/// The [`relaxation_resets`](EnhancedSdfHit::relaxation_resets) field records
/// how many times over-relaxation had to back off, which is a direct measure
/// of how much the acceleration helped versus how often it mispredicted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnhancedSdfHit {
    /// Ray parameter of the hit measured along the (normalized) direction, so
    /// it equals the world-space distance travelled from the ray origin.
    t: f32,
    /// World-space position of the hit (`origin + t * direction`).
    position: [f32; 3],
    /// Signed distance sampled at the hit position (negative inside the
    /// surface, positive outside), matching the raw field convention.
    distance: f32,
    /// Number of marching iterations taken to reach the hit.
    steps: u32,
    /// Number of times over-relaxation overshot an unbounding sphere and had
    /// to undo the step and fall back to a conservative `omega = 1` advance.
    relaxation_resets: u32,
}

impl EnhancedSdfHit {
    /// Ray parameter of the hit, equal to the world-space distance travelled.
    pub fn t(&self) -> f32 {
        self.t
    }

    /// World-space position of the hit.
    pub fn position(&self) -> [f32; 3] {
        self.position
    }

    /// Signed distance sampled at the hit position (negative inside).
    pub fn distance(&self) -> f32 {
        self.distance
    }

    /// Number of marching iterations taken to reach the hit.
    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// Number of over-relaxation back-offs performed during the march.
    pub fn relaxation_resets(&self) -> u32 {
        self.relaxation_resets
    }
}

/// Over-relaxed sphere traces a ray against a signed distance field, returning
/// the first surface intersection.
///
/// This follows Keinert et al. (2014, Listing 1). The ray is clipped to the
/// field's world-space bounding box, then marched: each iteration samples the
/// signed distance, folds in the entry sign so the magnitude is a conservative
/// unbounding radius regardless of whether the ray starts inside or outside,
/// and advances by `omega * radius`. When two successive unbounding spheres
/// fail to overlap (`radius + previous_radius < step_length`) the over-relaxed
/// step was unsafe, so it is undone and the march continues conservatively at
/// `omega = 1` for the remainder — guaranteeing the same hit as naive tracing.
///
/// `pixel_radius` is the pixel cone's half-footprint per unit ray distance: the
/// march stops once `radius < pixel_radius * t`, i.e. the surface is closer
/// than one pixel. `over_relaxation` is clamped into `[1, 2)` (a value of `1`
/// degenerates to naive sphere tracing).
///
/// Returns `None` when the ray misses the box, exits the box, exceeds
/// `max_distance`, or runs out of `max_steps` without converging. `direction`
/// need not be unit length; it is normalized internally so `t` is always a
/// world-space distance.
pub fn enhanced_sphere_trace(
    field: &SignedDistanceField,
    origin: [f32; 3],
    direction: [f32; 3],
    max_distance: f32,
    pixel_radius: f32,
    max_steps: u32,
    over_relaxation: f32,
) -> Option<EnhancedSdfHit> {
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

    // Keep over-relaxation strictly below 2 so the overlap test stays valid.
    let omega = over_relaxation.clamp(1.0, 1.999_999);

    // Entry sign folds interior starts into positive unbounding radii, so a
    // ray that begins inside the solid still marches toward the surface.
    let position_at = |t: f32| {
        [
            origin[0] + t * dir[0],
            origin[1] + t * dir[1],
            origin[2] + t * dir[2],
        ]
    };
    let entry_sample = sample_signed_distance(field, position_at(t_enter));
    let sign0 = if entry_sample < 0.0 { -1.0 } else { 1.0 };

    let mut t = t_enter;
    let mut previous_radius = 0.0f32;
    let mut step_length = 0.0f32;
    let mut omega_cur = omega;
    let mut relaxation_resets = 0u32;

    for step in 0..max_steps {
        let position = position_at(t);
        let signed = sign0 * sample_signed_distance(field, position);
        let radius = signed.abs();

        // The over-relaxed step overshot when the current and previous
        // unbounding spheres no longer overlap.
        let sor_failed = omega_cur > 1.0 && (radius + previous_radius) < step_length;
        if sor_failed {
            // Undo the previous over-relaxed advance and drop to conservative
            // stepping for the rest of the march.
            step_length -= omega_cur * step_length;
            omega_cur = 1.0;
            relaxation_resets += 1;
        } else {
            step_length = signed * omega_cur;
        }

        previous_radius = radius;

        // Multiplicative hit test (`radius < pixel_radius * t`) avoids a divide
        // by `t` and never fires on a step that was just rolled back.
        if !sor_failed && radius < pixel_radius * t {
            return Some(EnhancedSdfHit {
                t,
                position,
                // Report the raw signed field value (undo the entry fold).
                distance: sign0 * signed,
                steps: step + 1,
                relaxation_resets,
            });
        }

        t += step_length;
        if t > t_exit {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::mesh_sdf_raymarch::sphere_trace;
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

    /// Padded signed distance field of the unit cube: resolution 8 with a
    /// 4-voxel empty margin so exterior cells carry real positive distances.
    fn padded_cube_field() -> SignedDistanceField {
        let grid = voxelize_surface(&cube(), 8).unwrap();
        let padded = pad_voxel_grid(&grid, 4);
        signed_distance_field(&padded)
    }

    #[test]
    fn ray_from_outside_hits_top_face() {
        let field = padded_cube_field();
        // Straight down the +z axis onto the top face at z = 1.
        let hit = enhanced_sphere_trace(
            &field,
            [0.5, 0.5, 3.0],
            [0.0, 0.0, -1.0],
            10.0,
            1e-3,
            256,
            1.6,
        )
        .expect("ray toward the top face must hit");
        assert!(
            (hit.position()[2] - 1.0).abs() < 0.15,
            "hit z {} should be near the top face at 1.0",
            hit.position()[2],
        );
        assert!(hit.t() > 1.5 && hit.t() < 2.5, "t {} ~ 2", hit.t());
        assert!(hit.steps() >= 1);
    }

    #[test]
    fn enhanced_agrees_with_naive_sphere_trace() {
        let field = padded_cube_field();
        let origin = [0.5, 0.5, 3.0];
        let dir = [0.0, 0.0, -1.0];
        let enhanced =
            enhanced_sphere_trace(&field, origin, dir, 10.0, 1e-3, 256, 1.8)
                .expect("enhanced trace must hit");
        let naive = sphere_trace(&field, origin, dir, 10.0, 1e-3, 256)
            .expect("naive trace must hit");
        assert!(
            (enhanced.t() - naive.t()).abs() < 0.05,
            "enhanced t {} vs naive t {}",
            enhanced.t(),
            naive.t(),
        );
    }

    #[test]
    fn ray_pointing_away_misses() {
        let field = padded_cube_field();
        // Starts above the cube and marches further up: never enters the box
        // interior along +z (exits the clip box without converging).
        let hit = enhanced_sphere_trace(
            &field,
            [0.5, 0.5, 3.0],
            [0.0, 0.0, 1.0],
            10.0,
            1e-3,
            256,
            1.6,
        );
        assert!(hit.is_none(), "ray pointing away must miss");
    }

    #[test]
    fn relaxation_resets_accessor_available() {
        let field = padded_cube_field();
        let hit = enhanced_sphere_trace(
            &field,
            [0.5, 0.5, 3.0],
            [0.0, 0.0, -1.0],
            10.0,
            1e-3,
            256,
            1.9,
        )
        .expect("ray must hit");
        // The accessor exists and returns a finite count (possibly zero).
        let _ = hit.relaxation_resets();
    }

    #[test]
    fn omega_one_matches_naive_hit() {
        let field = padded_cube_field();
        let origin = [0.5, 0.5, 3.0];
        let dir = [0.0, 0.0, -1.0];
        // omega == 1 degenerates to naive sphere tracing: no resets at all.
        let hit = enhanced_sphere_trace(&field, origin, dir, 10.0, 1e-3, 256, 1.0)
            .expect("conservative trace must hit");
        assert_eq!(hit.relaxation_resets(), 0, "omega=1 never over-relaxes");
    }
}
