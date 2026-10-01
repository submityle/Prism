//! Top-level two-phase HZB occlusion query that ties the whole chain together.
//!
//! The pieces built up by the sibling modules each cover one stage of the
//! Nanite-style occlusion test:
//!
//! * [`project_world_aabb`](crate::project_world_aabb) maps world bounds to a
//!   screen-space [`HzbFootprint`](crate::HzbFootprint) and the candidate's
//!   nearest reverse-Z depth,
//! * [`HzbFootprint::sample_mip`](crate::HzbFootprint::sample_mip) picks the
//!   covering mip,
//! * [`HzbPyramid::gather_occluder`](crate::HzbPyramid::gather_occluder) reads
//!   the conservative occluder depth from the depth pyramid, and
//! * [`HzbTest::is_occluded`](crate::HzbTest) applies the reverse-Z comparison
//!   with its camera-cut / history / motion guards.
//!
//! [`test_bounds_occluded`] runs all four against a real pyramid and returns an
//! `Option<bool>` shaped to feed [`classify_early_hzb`](crate::classify_early_hzb)
//! and [`resolve_current_hzb`](crate::resolve_current_hzb) directly:
//!
//! * `Some(true)` — proven occluded, safe to reject,
//! * `Some(false)` — tested and visible,
//! * `None` — the test could not be evaluated (bounds behind the near plane,
//!   degenerate viewport, or no occluder in the pyramid), so the candidate is
//!   kept visible.

use crate::{
    project_world_aabb, GpuViewRecord, HzbPhase, HzbPyramid, HzbTest, WorldAabb,
};

/// Per-candidate tuning for a [`test_bounds_occluded`] query. The depth and
/// motion fields match [`HzbTest`](crate::HzbTest); the projected and sampled
/// depths plus the mip are computed by the query itself.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OcclusionQuery {
    /// Conservative depth slop added to the candidate's nearest depth before
    /// the comparison (`abs` is taken, matching [`HzbTest`](crate::HzbTest)).
    pub depth_bias: f32,
    /// Screen-space velocity magnitude used to widen the test under motion.
    pub projected_velocity: f32,
    /// The candidate's recorded history epoch (previous-phase validity check).
    pub history_epoch: u64,
    /// The epoch the view expects; a mismatch keeps the candidate visible in
    /// the previous phase.
    pub expected_history_epoch: u64,
}

/// Runs the full two-phase HZB occlusion test for a world-space AABB against a
/// reverse-Z depth pyramid.
///
/// Returns `Some(true)` when rejection is proven safe, `Some(false)` when the
/// candidate was tested and survives, and `None` when the test could not be
/// evaluated (bounds crossing the near plane, a degenerate viewport, or a
/// pyramid with no usable occluder). `None` and `Some(false)` both keep the
/// candidate visible and are interchangeable for
/// [`classify_early_hzb`](crate::classify_early_hzb) /
/// [`resolve_current_hzb`](crate::resolve_current_hzb).
pub fn test_bounds_occluded(
    view: &GpuViewRecord,
    pyramid: &HzbPyramid<'_>,
    aabb: WorldAabb,
    phase: HzbPhase,
    query: OcclusionQuery,
) -> Option<bool> {
    let projected = project_world_aabb(view, aabb)?;
    let sampled_mip_count = pyramid.mip_count();
    let mip = projected.footprint.sample_mip(sampled_mip_count);
    let occluder_depth = pyramid.gather_occluder(projected.footprint)?;

    let test = HzbTest {
        nearest_depth: projected.nearest_depth,
        occluder_depth,
        depth_bias: query.depth_bias,
        projected_velocity: query.projected_velocity,
        mip,
        sampled_mip_count,
        history_epoch: query.history_epoch,
        expected_history_epoch: query.expected_history_epoch,
        view_flags: view.flags,
    };
    Some(test.is_occluded(phase))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HzbMip, ViewFlags, ViewHandle};

    /// Reversed-Z clip matrix (column-major): `clip.w = z`, `clip.z = 1`, so
    /// `ndc.z = 1 / z` — nearer points read a larger reversed-Z depth.
    fn reverse_z_clip() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    }

    fn view() -> GpuViewRecord {
        GpuViewRecord {
            handle: ViewHandle {
                index: 0,
                generation: 1,
            },
            clip_from_world: reverse_z_clip(),
            previous_clip_from_world: reverse_z_clip(),
            world_position: [0.0, 0.0, 0.0],
            lod_scale: 1.0,
            viewport: [0, 0, 100, 100],
            frustum_planes: [[0.0, 0.0, 0.0, 1.0]; 6],
            layer_mask: 1,
            flags: ViewFlags::REVERSE_Z,
            history_epoch: 7,
        }
    }

    fn query() -> OcclusionQuery {
        OcclusionQuery {
            depth_bias: 0.0,
            projected_velocity: 0.0,
            history_epoch: 7,
            expected_history_epoch: 7,
        }
    }

    // A full-screen single-mip pyramid where every texel reads a near depth
    // (0.9 reversed-Z, i.e. an occluder close to the camera at z ~= 1.11).
    fn near_wall_pyramid(buffer: &[f32]) -> [HzbMip<'_>; 1] {
        [HzbMip {
            width: 100,
            height: 100,
            texels: buffer,
        }]
    }

    #[test]
    fn far_candidate_behind_a_near_wall_is_rejected() {
        let buffer = [0.9_f32; 100 * 100];
        let mips = near_wall_pyramid(&buffer);
        let pyramid = HzbPyramid::new(&mips);
        // A small candidate far away at z = 10 -> nearest reversed-Z depth 0.1,
        // which is behind the 0.9 wall: proven occluded.
        let aabb = WorldAabb::from_center_half_extents([0.0, 0.0, 10.0], [0.2, 0.2, 0.0]);
        assert_eq!(
            test_bounds_occluded(&view(), &pyramid, aabb, HzbPhase::Current, query()),
            Some(true)
        );
    }

    #[test]
    fn near_candidate_in_front_of_the_wall_survives() {
        let buffer = [0.9_f32; 100 * 100];
        let mips = near_wall_pyramid(&buffer);
        let pyramid = HzbPyramid::new(&mips);
        // A candidate at z = 0.5 -> nearest reversed-Z depth 2.0, well in front
        // of the 0.9 wall: tested and visible.
        let aabb = WorldAabb::from_center_half_extents([0.0, 0.0, 0.5], [0.05, 0.05, 0.0]);
        assert_eq!(
            test_bounds_occluded(&view(), &pyramid, aabb, HzbPhase::Current, query()),
            Some(false)
        );
    }

    #[test]
    fn bounds_crossing_the_near_plane_cannot_be_tested() {
        let buffer = [0.9_f32; 100 * 100];
        let mips = near_wall_pyramid(&buffer);
        let pyramid = HzbPyramid::new(&mips);
        let aabb = WorldAabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert_eq!(
            test_bounds_occluded(&view(), &pyramid, aabb, HzbPhase::Current, query()),
            None
        );
    }

    #[test]
    fn empty_pyramid_cannot_be_tested() {
        let pyramid = HzbPyramid::new(&[]);
        let aabb = WorldAabb::from_center_half_extents([0.0, 0.0, 10.0], [0.2, 0.2, 0.0]);
        assert_eq!(
            test_bounds_occluded(&view(), &pyramid, aabb, HzbPhase::Current, query()),
            None
        );
    }

    #[test]
    fn camera_cut_keeps_a_would_be_occluded_candidate_visible() {
        let buffer = [0.9_f32; 100 * 100];
        let mips = near_wall_pyramid(&buffer);
        let pyramid = HzbPyramid::new(&mips);
        let mut v = view();
        v.flags = ViewFlags::REVERSE_Z | ViewFlags::CAMERA_CUT;
        // Same geometry that was rejected above now survives: the camera-cut
        // guard inside HzbTest forbids rejection on an invalid history.
        let aabb = WorldAabb::from_center_half_extents([0.0, 0.0, 10.0], [0.2, 0.2, 0.0]);
        assert_eq!(
            test_bounds_occluded(&v, &pyramid, aabb, HzbPhase::Current, query()),
            Some(false)
        );
    }

    #[test]
    fn previous_phase_rejects_stale_history_before_current_phase_accepts() {
        let buffer = [0.9_f32; 100 * 100];
        let mips = near_wall_pyramid(&buffer);
        let pyramid = HzbPyramid::new(&mips);
        let aabb = WorldAabb::from_center_half_extents([0.0, 0.0, 10.0], [0.2, 0.2, 0.0]);
        // Stale history: the previous phase must not reject, but the current
        // phase (which ignores history epochs) still proves occlusion.
        let stale = OcclusionQuery {
            expected_history_epoch: 999,
            ..query()
        };
        assert_eq!(
            test_bounds_occluded(&view(), &pyramid, aabb, HzbPhase::Previous, stale),
            Some(false)
        );
        assert_eq!(
            test_bounds_occluded(&view(), &pyramid, aabb, HzbPhase::Current, stale),
            Some(true)
        );
    }
}
