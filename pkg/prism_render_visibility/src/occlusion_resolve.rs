//! Real HZB-driven occlusion resolution.
//!
//! [`cull_view`](crate::cull_view) consumes a pre-computed
//! `occluded: &BTreeSet<(ViewHandle, SceneHandle)>` set to reject hidden
//! instances. Historically that set was supplied externally (e.g. a look-up
//! table), which made the two-phase HZB chain a half-wired reference: the
//! projection, pyramid gather, and reverse-Z test existed but nothing fed their
//! verdicts back into culling.
//!
//! [`resolve_occluded_set`] closes that loop. It walks the same instance
//! handles `cull_view` will visit, reads each instance's world-space bounds
//! straight from the [`CpuRenderScene`], and runs the full
//! [`test_bounds_occluded`](crate::test_bounds_occluded) chain against a real
//! reverse-Z depth [`HzbPyramid`](crate::HzbPyramid). The returned set has
//! exactly the shape `VisibilityInput::occluded` expects, so the result can be
//! handed to `cull_view` directly with no adapter.
//!
//! The resolver is deliberately conservative: only `Some(true)` (rejection
//! proven safe) inserts a pair. Instances behind the near plane, in a
//! degenerate viewport, with no occluder in the pyramid, under a camera cut, or
//! with stale history all stay visible, matching the guards baked into
//! [`HzbTest`](crate::HzbTest). Handles that no longer resolve in the scene are
//! skipped the same way `cull_view` skips stale handles.

use crate::{
    test_bounds_occluded, GpuViewRecord, HzbPhase, HzbPyramid, OcclusionQuery, ViewHandle,
    WorldAabb,
};
use alloc::collections::BTreeSet;
use prism_render_architecture::gpu_scene::{CpuRenderScene, SceneHandle};

/// Computes the `(view, instance)` pairs that the two-phase HZB chain proves
/// occluded for `view`, ready to drop into
/// [`VisibilityInput::occluded`](crate::VisibilityInput).
///
/// For every handle in `handles` the instance's world bounds
/// (`center` ± `half_extents`) are projected and tested against `pyramid` via
/// [`test_bounds_occluded`](crate::test_bounds_occluded) in `phase`, reusing
/// `base_query` for the depth-bias / motion / history fields. A pair is
/// inserted only when the test returns `Some(true)`; `Some(false)` and `None`
/// keep the instance visible. Handles missing from `scene` are skipped.
///
/// The traversal is linear in `handles` and allocates only the output set, so
/// it can run per view each frame as the real source of the culling occlusion
/// mask instead of a pre-baked table.
pub fn resolve_occluded_set(
    view: &GpuViewRecord,
    pyramid: &HzbPyramid<'_>,
    scene: &CpuRenderScene,
    handles: &[SceneHandle],
    phase: HzbPhase,
    base_query: OcclusionQuery,
) -> BTreeSet<(ViewHandle, SceneHandle)> {
    let mut occluded = BTreeSet::new();
    for &handle in handles {
        let Some(instance) = scene.get(handle) else {
            continue;
        };
        let aabb = WorldAabb::from_center_half_extents(
            instance.bounds.center,
            instance.bounds.half_extents,
        );
        if test_bounds_occluded(view, pyramid, aabb, phase, base_query) == Some(true) {
            occluded.insert((view.handle, handle));
        }
    }
    occluded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HzbMip, ViewFlags};
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{
        CpuRenderScene, InstanceRecord, SceneBounds, SceneOperation, SceneTransaction,
    };

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
            handle: GenerationalHandle {
                index: 3,
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

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle {
            index,
            generation: 1,
        }
    }

    /// Builds a scene with one instance per `(handle, center)` pair, each a
    /// small AABB (`half_extents` 0.2) so projection stays on-screen.
    fn scene_with(instances: &[(GenerationalHandle, [f32; 3])]) -> CpuRenderScene {
        let mut scene = CpuRenderScene::default();
        let operations = instances
            .iter()
            .map(|&(handle, center)| SceneOperation::Create {
                handle,
                record: InstanceRecord {
                    bounds: SceneBounds {
                        center,
                        radius: 0.4,
                        half_extents: [0.2, 0.2, 0.0],
                        _padding: 0.0,
                    },
                    render_layers: 1,
                    ..Default::default()
                },
            })
            .collect();
        scene.apply(&SceneTransaction {
            frame_epoch: 1,
            sequence: 1,
            producer: 1,
            operations,
        });
        scene
    }

    // A full-screen single-mip pyramid where every texel reads a near depth
    // (0.9 reversed-Z): an occluder close to the camera.
    fn near_wall_buffer() -> [f32; 100 * 100] {
        [0.9_f32; 100 * 100]
    }

    #[test]
    fn only_instances_behind_the_wall_are_resolved_occluded() {
        let buffer = near_wall_buffer();
        let mips = [HzbMip {
            width: 100,
            height: 100,
            texels: &buffer,
        }];
        let pyramid = HzbPyramid::new(&mips);
        // index 1: far at z = 10 -> reversed-Z 0.1, behind the 0.9 wall.
        // index 2: near at z = 0.5 -> reversed-Z 2.0, in front of the wall.
        let far = handle(1);
        let near = handle(2);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0]), (near, [0.0, 0.0, 0.5])]);
        let occluded = resolve_occluded_set(
            &view(),
            &pyramid,
            &scene,
            &[far, near],
            HzbPhase::Current,
            query(),
        );
        assert!(occluded.contains(&(view().handle, far)));
        assert!(!occluded.contains(&(view().handle, near)));
        assert_eq!(occluded.len(), 1);
    }

    #[test]
    fn empty_pyramid_resolves_nothing() {
        let pyramid = HzbPyramid::new(&[]);
        let far = handle(1);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0])]);
        let occluded = resolve_occluded_set(
            &view(),
            &pyramid,
            &scene,
            &[far],
            HzbPhase::Current,
            query(),
        );
        assert!(occluded.is_empty());
    }

    #[test]
    fn camera_cut_keeps_everything_visible() {
        let buffer = near_wall_buffer();
        let mips = [HzbMip {
            width: 100,
            height: 100,
            texels: &buffer,
        }];
        let pyramid = HzbPyramid::new(&mips);
        let mut v = view();
        v.flags = ViewFlags::REVERSE_Z | ViewFlags::CAMERA_CUT;
        let far = handle(1);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0])]);
        let occluded =
            resolve_occluded_set(&v, &pyramid, &scene, &[far], HzbPhase::Current, query());
        assert!(occluded.is_empty());
    }

    #[test]
    fn stale_handles_are_skipped_not_counted_occluded() {
        let buffer = near_wall_buffer();
        let mips = [HzbMip {
            width: 100,
            height: 100,
            texels: &buffer,
        }];
        let pyramid = HzbPyramid::new(&mips);
        let far = handle(1);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0])]);
        // A handle whose generation does not match any live slot resolves to
        // None in the scene and must be skipped rather than treated occluded.
        let ghost = GenerationalHandle {
            index: 1,
            generation: 99,
        };
        let occluded = resolve_occluded_set(
            &view(),
            &pyramid,
            &scene,
            &[ghost],
            HzbPhase::Current,
            query(),
        );
        assert!(occluded.is_empty());
    }
}
