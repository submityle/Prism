//! Per-frame screen-coverage estimation that drives the cloth LOD gate.
//!
//! The architecture layer (`prism_render_architecture::cloth::lod`) deliberately
//! does not do any projection math: it classifies a piece into a tier given a
//! coverage scalar the *caller* supplies. This module is that caller. It turns a
//! garment's live world-space particle cloud into a projected screen-space area
//! fraction each frame, so the LOD gate in [`prepare`](super::prepare) stops
//! being a static authored knob and starts reacting to how large the garment
//! actually is on screen.
//!
//! Two halves live here, split by testability:
//!
//! * [`garment_world_aabb`] and [`screen_coverage`] are pure, deterministic
//!   functions over `glam` types. They own the projection math and are unit
//!   tested directly against hand-built matrices, with no `bevy` world in sight.
//! * [`update_cloth_coverage`] is the main-world system that reads the active
//!   camera, composes its `clip_from_world` matrix and writes the resolved
//!   coverage back onto every [`ClothGarment`]. It runs in `PostUpdate` after
//!   the camera projection is refreshed and before the extract stage snapshots
//!   the garments into the render world.
//!
//! The coverage metric is the standard "projected screen bounding box" fraction:
//! the eight corners of the garment's world AABB are projected to normalized
//! device coordinates, the resulting 2D bounding rectangle is clipped to the
//! `[-1, 1]` viewport and its area is divided by the full `2x2 = 4` NDC area.
//! The result is a monotonic `0..=1` proxy for apparent on-screen size, matching
//! the UE-style screen-size heuristic closely enough to drive tier selection
//! while staying trivially deterministic and testable.

use bevy_camera::Camera;
use bevy_ecs::prelude::*;
use bevy_math::{Mat4, Vec3};
use bevy_transform::components::GlobalTransform;

use super::garment::ClothGarment;

/// Clip-space `w` below which a projected corner is treated as on or behind the
/// near plane. A box with any such corner is straddling the camera and is, by
/// definition, filling the view, so coverage saturates to `1.0` rather than
/// dividing by a near-zero (or negative) `w` and producing a flipped NDC box.
const NEAR_PLANE_W_EPSILON: f32 = 1.0e-4;

/// Computes the world-space axis-aligned bounding box of a garment's particle
/// cloud.
///
/// Particle positions pack the inverse mass into `.w`; only `xyz` contribute to
/// the spatial bounds. Returns [`None`] for an empty cloud so the caller can
/// skip a garment that has no geometry this frame (its coverage is meaningless
/// and must not be overwritten).
#[must_use]
pub(crate) fn garment_world_aabb(positions: &[[f32; 4]]) -> Option<(Vec3, Vec3)> {
    let (first, rest) = positions.split_first()?;
    let mut min = Vec3::new(first[0], first[1], first[2]);
    let mut max = min;
    for position in rest {
        let point = Vec3::new(position[0], position[1], position[2]);
        min = min.min(point);
        max = max.max(point);
    }
    Some((min, max))
}

/// Projects a world-space AABB through `clip_from_world` and returns the
/// fraction of the viewport its screen-space bounding rectangle covers, in
/// `0..=1`.
///
/// The eight corners are projected to clip space; if any corner lies on or
/// behind the near plane the box straddles the camera and coverage saturates to
/// `1.0`. Otherwise the perspective-divided NDC bounding rectangle is clipped to
/// the `[-1, 1]` viewport and its area is normalized by the full `4.0` NDC area.
/// A box entirely outside the viewport (or a degenerate zero-extent box) yields
/// `0.0`.
#[must_use]
pub(crate) fn screen_coverage(clip_from_world: Mat4, aabb_min: Vec3, aabb_max: Vec3) -> f32 {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { aabb_min.x } else { aabb_max.x },
            if i & 2 == 0 { aabb_min.y } else { aabb_max.y },
            if i & 4 == 0 { aabb_min.z } else { aabb_max.z },
        );
        let clip = clip_from_world * corner.extend(1.0);
        if clip.w <= NEAR_PLANE_W_EPSILON {
            // The box reaches the near plane (or wraps behind the camera): it is
            // as close as geometry can get, so it fills the view.
            return 1.0;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        min_x = min_x.min(ndc_x);
        min_y = min_y.min(ndc_y);
        max_x = max_x.max(ndc_x);
        max_y = max_y.max(ndc_y);
    }

    // Clip the projected rectangle to the visible NDC viewport.
    let clipped_min_x = min_x.max(-1.0);
    let clipped_max_x = max_x.min(1.0);
    let clipped_min_y = min_y.max(-1.0);
    let clipped_max_y = max_y.min(1.0);

    let width = clipped_max_x - clipped_min_x;
    let height = clipped_max_y - clipped_min_y;
    if width <= 0.0 || height <= 0.0 {
        // Entirely off-screen, or a degenerate box that projects to a line or
        // point: it occupies no visible area.
        return 0.0;
    }

    // The NDC viewport spans `[-1, 1]` on each axis, an area of `4.0`.
    (width * height / 4.0).clamp(0.0, 1.0)
}

/// Composes the active camera's `clip_from_world` matrix.
///
/// This is `clip_from_view * view_from_world`, where `view_from_world` is the
/// inverse of the camera's world transform. The projection matrix comes from the
/// camera's `computed` values, which `PostUpdate`'s camera system refreshes
/// before this system runs.
#[must_use]
fn clip_from_world(camera: &Camera, camera_transform: &GlobalTransform) -> Mat4 {
    camera.clip_from_view() * camera_transform.to_matrix().inverse()
}

/// Writes each garment's projected screen coverage for the frame.
///
/// Selects the active camera with the highest render order (the same
/// "topmost active" rule `bevy` uses to pick the primary view), projects every
/// garment's world AABB through its `clip_from_world` and stores the resulting
/// coverage back on the garment so the extract stage carries it into the LOD
/// gate. Garments with no particles keep their prior coverage untouched, and the
/// write is skipped when the coverage is unchanged so idle garments do not
/// churn change detection.
///
/// When no active camera exists the system is a no-op: with nothing to project
/// against, the last known coverage is the best estimate and overwriting it with
/// a guess would be dishonest.
pub(crate) fn update_cloth_coverage(
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut garments: Query<&mut ClothGarment>,
) {
    let Some((camera, camera_transform)) = cameras
        .iter()
        .filter(|(camera, _)| camera.is_active)
        .max_by_key(|(camera, _)| camera.order)
    else {
        return;
    };

    let clip_from_world = clip_from_world(camera, camera_transform);

    for mut garment in &mut garments {
        let Some((min, max)) = garment_world_aabb(garment.positions()) else {
            continue;
        };
        let coverage = screen_coverage(clip_from_world, min, max);
        if (garment.coverage() - coverage).abs() > f32::EPSILON {
            garment.set_coverage(coverage);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec4;

    /// A minimal right-handed pinhole projection: the camera looks down `-Z`, so
    /// a point's clip-space `w` is `-z` (positive in front of the camera,
    /// negative behind it) and its NDC `xy` is `xy / -z`. Built by hand so the
    /// projection math is exercised without any deprecated `glam` constructor.
    fn pinhole() -> Mat4 {
        Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, -1.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    /// A box that exactly fills the `[-1, 1]` NDC viewport under an identity
    /// projection covers the whole screen.
    #[test]
    fn full_viewport_box_covers_everything() {
        let coverage = screen_coverage(
            Mat4::IDENTITY,
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );
        assert!((coverage - 1.0).abs() <= 1.0e-6, "coverage = {coverage}");
    }

    /// A box spanning half the viewport on each axis covers a quarter of the
    /// screen area.
    #[test]
    fn half_extent_box_covers_a_quarter() {
        let coverage = screen_coverage(
            Mat4::IDENTITY,
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
        );
        assert!((coverage - 0.25).abs() <= 1.0e-6, "coverage = {coverage}");
    }

    /// A box wholly outside the viewport contributes no coverage.
    #[test]
    fn offscreen_box_covers_nothing() {
        let coverage = screen_coverage(
            Mat4::IDENTITY,
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(3.0, 3.0, 0.0),
        );
        assert_eq!(coverage, 0.0);
    }

    /// A box straddling the viewport edge is clipped: only the visible quadrant
    /// counts.
    #[test]
    fn partially_offscreen_box_is_clipped() {
        // Spans [0, 2] on each axis; only [0, 1] x [0, 1] is visible -> one unit
        // square of the 4-unit NDC area.
        let coverage = screen_coverage(
            Mat4::IDENTITY,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
        );
        assert!((coverage - 0.25).abs() <= 1.0e-6, "coverage = {coverage}");
    }

    /// A degenerate zero-extent box (a single point) projects to a point and
    /// covers no area.
    #[test]
    fn degenerate_point_box_covers_nothing() {
        let coverage = screen_coverage(
            Mat4::IDENTITY,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        );
        assert_eq!(coverage, 0.0);
    }

    /// A perspective box that reaches through the near plane saturates coverage
    /// rather than dividing by a near-zero `w`.
    #[test]
    fn box_straddling_the_near_plane_saturates() {
        // The box spans z in [-1, 1]: its front corners are ahead of the camera
        // and its back corners are behind it, so at least one corner has w <= 0.
        let coverage = screen_coverage(
            pinhole(),
            Vec3::new(-0.5, -0.5, -1.0),
            Vec3::new(0.5, 0.5, 1.0),
        );
        assert_eq!(coverage, 1.0);
    }

    /// Under perspective, a closer box covers strictly more screen area than the
    /// same box farther away, and both stay within `(0, 1)`.
    #[test]
    fn perspective_coverage_grows_as_the_box_approaches() {
        let proj = pinhole();
        let near = screen_coverage(
            proj,
            Vec3::new(-0.5, -0.5, -3.0),
            Vec3::new(0.5, 0.5, -2.0),
        );
        let far = screen_coverage(
            proj,
            Vec3::new(-0.5, -0.5, -11.0),
            Vec3::new(0.5, 0.5, -10.0),
        );
        assert!(near > far, "near {near} should exceed far {far}");
        assert!(far > 0.0 && near < 1.0, "near {near}, far {far}");
    }

    /// The world AABB spans the extremes of the particle cloud and ignores the
    /// inverse-mass `.w` component.
    #[test]
    fn world_aabb_spans_the_particle_cloud() {
        let positions = [
            [-1.0, 2.0, 0.5, 0.0],
            [3.0, -4.0, 0.5, 1.0],
            [0.0, 0.0, 7.0, 2.0],
        ];
        let (min, max) = garment_world_aabb(&positions).expect("non-empty cloud");
        assert_eq!(min, Vec3::new(-1.0, -4.0, 0.5));
        assert_eq!(max, Vec3::new(3.0, 2.0, 7.0));
    }

    /// An empty particle cloud has no bounds, so the caller must skip it rather
    /// than fabricate a coverage.
    #[test]
    fn world_aabb_of_empty_cloud_is_none() {
        assert!(garment_world_aabb(&[]).is_none());
    }
}
