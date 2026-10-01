//! Projects world-space bounds into a screen-space [`HzbFootprint`] for the
//! two-phase HZB occlusion test.
//!
//! This closes the gap between a candidate's world-space bounds and the
//! footprint math in [`hzb_footprint`](crate::HzbFootprint): it transforms the
//! eight corners of a world-space AABB through the view's
//! [`clip_from_world`](crate::GpuViewRecord::clip_from_world) matrix, performs
//! the perspective divide, and maps the resulting NDC into mip-0 HZB texels.
//! The output pairs the pixel-space [`HzbFootprint`] (which drives
//! [`HzbFootprint::sample_mip`](crate::HzbFootprint::sample_mip)) with the
//! candidate's nearest reverse-Z depth (which fills
//! [`HzbTest::nearest_depth`](crate::HzbTest)).
//!
//! The projection reuses the shipping meshlet software-raster vertex path
//! [`project_vertex`](prism_render_architecture::virtual_geometry::project_vertex)
//! so the CPU occlusion footprint matches the GPU vertex transform bit-for-bit
//! (column-major `clip_from_world`, y-down pixel space, reversed-Z depth with
//! `1.0` nearest).
//!
//! # Conservatism
//!
//! Occlusion culling must never reject a visible candidate. Two guards keep
//! the test safe:
//!
//! * If **any** corner lies on or behind the camera near plane
//!   (`clip.w <= 0`, where the perspective divide is undefined), the whole
//!   projection is abandoned and [`None`] is returned, so the caller keeps the
//!   candidate visible rather than testing a garbage footprint.
//! * The footprint is **not** clamped to the viewport. An AABB that pokes
//!   off-screen yields a larger footprint, which only ever selects a *coarser*
//!   mip whose min-reduction returns a *farther* occluder — strictly more
//!   conservative, never a wrongful rejection.

use crate::{GpuViewRecord, HzbFootprint};
use prism_render_architecture::virtual_geometry::project_vertex;

/// A world-space axis-aligned bounding box given by its minimum and maximum
/// corners (`[x, y, z]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldAabb {
    /// Minimum corner (`[x, y, z]`).
    pub min: [f32; 3],
    /// Maximum corner (`[x, y, z]`).
    pub max: [f32; 3],
}

impl WorldAabb {
    /// Builds an AABB from its minimum and maximum corners.
    pub const fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Builds an AABB from a center and non-negative half-extents.
    pub fn from_center_half_extents(center: [f32; 3], half_extents: [f32; 3]) -> Self {
        let hx = half_extents[0].abs();
        let hy = half_extents[1].abs();
        let hz = half_extents[2].abs();
        Self {
            min: [center[0] - hx, center[1] - hy, center[2] - hz],
            max: [center[0] + hx, center[1] + hy, center[2] + hz],
        }
    }

    /// The eight world-space corners, in a fixed order (every combination of
    /// min/max on each axis).
    pub fn corners(self) -> [[f32; 3]; 8] {
        let [lx, ly, lz] = self.min;
        let [hx, hy, hz] = self.max;
        [
            [lx, ly, lz],
            [hx, ly, lz],
            [lx, hy, lz],
            [hx, hy, lz],
            [lx, ly, hz],
            [hx, ly, hz],
            [lx, hy, hz],
            [hx, hy, hz],
        ]
    }
}

/// The screen-space result of projecting a [`WorldAabb`] for an HZB test: the
/// pixel-space [`HzbFootprint`] plus the candidate's nearest reverse-Z depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedBounds {
    /// Pixel-space (mip-0 texel) footprint of the projected bounds.
    pub footprint: HzbFootprint,
    /// Nearest reverse-Z depth over all corners (the maximum NDC `z`, since
    /// `1.0` is nearest). Feeds [`HzbTest::nearest_depth`](crate::HzbTest).
    pub nearest_depth: f32,
}

/// Projects a world-space AABB into an HZB [`ProjectedBounds`] for `view`.
///
/// Returns [`None`] — meaning "cannot occlusion-test, keep visible" — when the
/// view has a degenerate viewport, when any corner is on or behind the near
/// plane, or when any projected coordinate is non-finite. Otherwise the
/// returned footprint is the pixel-space bounding box of the eight projected
/// corners and `nearest_depth` is their farthest-forward reverse-Z depth.
pub fn project_world_aabb(view: &GpuViewRecord, aabb: WorldAabb) -> Option<ProjectedBounds> {
    let width = view.viewport[2];
    let height = view.viewport[3];
    if width == 0 || height == 0 {
        return None;
    }
    let viewport = [width as f32, height as f32];

    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    let mut nearest_depth = f32::NEG_INFINITY;

    for corner in aabb.corners() {
        // A single corner behind the near plane makes the whole bound straddle
        // it; the HZB test is undefined there, so bail conservatively.
        let projected = project_vertex(&view.clip_from_world, corner, viewport)?;
        let [px, py] = projected.pos;
        let depth = projected.depth;
        if !px.is_finite() || !py.is_finite() || !depth.is_finite() {
            return None;
        }
        if px < min[0] {
            min[0] = px;
        }
        if py < min[1] {
            min[1] = py;
        }
        if px > max[0] {
            max[0] = px;
        }
        if py > max[1] {
            max[1] = py;
        }
        // Reverse-Z: larger depth is nearer, so the max is the nearest point.
        if depth > nearest_depth {
            nearest_depth = depth;
        }
    }

    Some(ProjectedBounds {
        footprint: HzbFootprint::new(min, max),
        nearest_depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HzbPhase, HzbTest, ViewFlags, ViewHandle};

    /// A reversed-Z perspective-like clip matrix stored column-major.
    /// Column 3 (`[0, 0, 1, 0]`) copies world `z` into `clip.w`, so `w == z`
    /// (a point at world `z` sits that many units in front of the camera) and
    /// the perspective divide is well defined for `z > 0`. Row 2
    /// (`clip.z = 0 * x + 0 * y + 0 * z + 1`) gives a constant `clip.z = 1`, so
    /// `ndc.z = 1 / z`: nearer points (small `z`) read a larger reversed-Z
    /// depth, exactly as a real reversed-Z projection does.
    fn reverse_z_clip() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0], // column 0
            [0.0, 1.0, 0.0, 0.0], // column 1
            [0.0, 0.0, 0.0, 1.0], // column 2 -> clip.w = z
            [0.0, 0.0, 1.0, 0.0], // column 3 -> clip.z = 1
        ]
    }

    fn view(clip: [[f32; 4]; 4]) -> GpuViewRecord {
        GpuViewRecord {
            handle: ViewHandle {
                index: 0,
                generation: 1,
            },
            clip_from_world: clip,
            previous_clip_from_world: clip,
            world_position: [0.0, 0.0, 0.0],
            lod_scale: 1.0,
            viewport: [0, 0, 100, 100],
            frustum_planes: [[0.0, 0.0, 0.0, 1.0]; 6],
            layer_mask: 1,
            flags: ViewFlags::REVERSE_Z,
            history_epoch: 1,
        }
    }

    #[test]
    fn centered_box_projects_to_the_viewport_center() {
        // A small box centred on the view axis at z = 2 projects around the
        // viewport centre (50, 50) with a positive extent.
        let v = view(reverse_z_clip());
        let aabb = WorldAabb::from_center_half_extents([0.0, 0.0, 2.0], [0.1, 0.1, 0.0]);
        let projected = project_world_aabb(&v, aabb).expect("in front of camera");
        let [w, h] = projected.footprint.extent();
        assert!(w > 0.0 && h > 0.0, "nondegenerate footprint, got {w}x{h}");
        // The footprint straddles the viewport centre on both axes.
        assert!(projected.footprint.min[0] < 50.0 && projected.footprint.max[0] > 50.0);
        assert!(projected.footprint.min[1] < 50.0 && projected.footprint.max[1] > 50.0);
        // nearest reverse-Z depth = 1 / z_near = 1 / 2 = 0.5.
        assert!((projected.nearest_depth - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn nearer_box_reads_a_larger_reverse_z_depth() {
        let v = view(reverse_z_clip());
        let near = project_world_aabb(&v, WorldAabb::from_center_half_extents([0.0, 0.0, 1.0], [0.1; 3]))
            .expect("in front");
        let far = project_world_aabb(&v, WorldAabb::from_center_half_extents([0.0, 0.0, 4.0], [0.1; 3]))
            .expect("in front");
        // The nearer box (z spanning ~0.9..1.1) reads a larger reversed-Z
        // nearest depth than the far box (z ~3.9..4.1).
        assert!(near.nearest_depth > far.nearest_depth);
    }

    #[test]
    fn bounds_crossing_the_near_plane_cannot_be_tested() {
        let v = view(reverse_z_clip());
        // This AABB spans z = -1..1, so corners at z <= 0 sit on/behind the
        // camera: the projection must bail and keep the candidate visible.
        let straddling = WorldAabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert_eq!(project_world_aabb(&v, straddling), None);
    }

    #[test]
    fn degenerate_viewport_cannot_be_tested() {
        let mut v = view(reverse_z_clip());
        v.viewport = [0, 0, 0, 100];
        let aabb = WorldAabb::from_center_half_extents([0.0, 0.0, 2.0], [0.1; 3]);
        assert_eq!(project_world_aabb(&v, aabb), None);
    }

    #[test]
    fn projection_feeds_the_full_occlusion_chain() {
        // End-to-end: project a far candidate, select its mip, then test it
        // against a near occluder gather. Reverse-Z occlusion rejects the
        // candidate because its nearest point is behind the occluder.
        let v = view(reverse_z_clip());
        let candidate = project_world_aabb(
            &v,
            WorldAabb::from_center_half_extents([0.0, 0.0, 8.0], [2.0, 2.0, 0.0]),
        )
        .expect("in front");
        let mip = candidate.footprint.sample_mip(8);
        // The occluder gather sits nearer (depth ~0.5 = z 2) than the
        // candidate's nearest depth (~1/8).
        let occluder = crate::conservative_occluder_reverse_z(&[0.52, 0.50, 0.51, 0.49])
            .expect("finite gather");
        let test = HzbTest {
            nearest_depth: candidate.nearest_depth,
            occluder_depth: occluder,
            depth_bias: 0.0,
            projected_velocity: 0.0,
            mip,
            sampled_mip_count: 8,
            history_epoch: 1,
            expected_history_epoch: 1,
            view_flags: ViewFlags::REVERSE_Z,
        };
        assert!(test.nearest_depth < occluder, "candidate is behind the occluder");
        assert!(test.is_occluded(HzbPhase::Current));
    }
}
