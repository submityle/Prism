//! Scene-level two-phase HZB occlusion resolution (Nanite-style).
//!
//! [`resolve_occluded_set`](crate::resolve_occluded_set) runs a *single* HZB
//! test per instance, which is only safe against a depth hierarchy already
//! known to contain every occluder. Real engines cannot assume that: the
//! current frame's depth is not available until something has been drawn. UE's
//! Nanite answers this with a two-phase loop —
//!
//! 1. **Early phase**: test each candidate against the *previous* frame's HZB.
//!    Proven-occluded candidates are *deferred*, not culled, because last
//!    frame's depth may be stale.
//! 2. The early-visible set is rasterised, producing the *current* frame's
//!    depth, from which a fresh HZB is built
//!    ([`build_hzb_pyramid`](crate::build_hzb_pyramid)).
//! 3. **Late phase**: re-test only the deferred candidates against the current
//!    HZB. A candidate is culled *only* when it is occluded in this second
//!    test as well; otherwise it is revealed and drawn.
//!
//! [`resolve_two_phase_occlusion`] performs that loop over a scene, delegating
//! the per-stage bookkeeping to [`classify_early_hzb`](crate::classify_early_hzb)
//! and [`resolve_current_hzb`](crate::resolve_current_hzb) so the stage logic
//! stays in one place. The returned `occluded` set has exactly the shape
//! [`VisibilityInput::occluded`](crate::VisibilityInput) expects, so it is a
//! strictly more correct drop-in than the single-phase resolver: an instance is
//! rejected only when it is proven hidden in *both* the previous and current
//! depth hierarchies.

use crate::{
    classify_early_hzb, resolve_current_hzb, test_bounds_occluded, GpuViewRecord, HzbPhase,
    HzbPyramid, OcclusionQuery, ViewHandle, VisibilityStageMask, WorldAabb,
};
use alloc::collections::BTreeSet;
use prism_render_architecture::gpu_scene::{CpuRenderScene, SceneHandle};

/// Inputs to [`resolve_two_phase_occlusion`]: the view, both temporal depth
/// pyramids, the scene slice to test, and the shared occlusion query.
pub struct TwoPhaseHzbInput<'a> {
    /// The view whose flags, projection, and history gate the test.
    pub view: &'a GpuViewRecord,
    /// Previous frame's reverse-Z HZB, queried in the early phase.
    pub previous_pyramid: &'a HzbPyramid<'a>,
    /// Current frame's reverse-Z HZB, queried in the late phase.
    pub current_pyramid: &'a HzbPyramid<'a>,
    /// The scene whose instance bounds are projected and tested.
    pub scene: &'a CpuRenderScene,
    /// Instance handles to classify, in traversal order.
    pub handles: &'a [SceneHandle],
    /// Depth-bias / motion / history fields shared by both phase queries.
    pub base_query: OcclusionQuery,
    /// The view's previous-frame history epoch; `Reuse` history is required for
    /// any candidate to be deferred (and thus eligible for culling).
    pub previous_history_epoch: Option<u64>,
}

/// The verdict of a two-phase occlusion pass.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TwoPhaseOcclusion {
    /// Candidates occluded in the early phase and therefore deferred to the
    /// late phase (a superset of [`occluded`](TwoPhaseOcclusion::occluded)).
    pub late_retest: BTreeSet<(ViewHandle, SceneHandle)>,
    /// Candidates proven occluded in *both* phases, safe to cull. Ready to feed
    /// [`VisibilityInput::occluded`](crate::VisibilityInput).
    pub occluded: BTreeSet<(ViewHandle, SceneHandle)>,
}

/// Resolves the Nanite-style two-phase HZB occlusion verdict for a scene.
///
/// For every handle the instance's world bounds are projected and tested
/// against `previous_pyramid` in [`HzbPhase::Previous`](crate::HzbPhase); the
/// result is classified by [`classify_early_hzb`](crate::classify_early_hzb)
/// using the view's history policy. Deferred candidates
/// ([`LATE_RETEST`](crate::VisibilityStageMask::LATE_RETEST)) are re-tested
/// against `current_pyramid` in [`HzbPhase::Current`](crate::HzbPhase) and
/// resolved by [`resolve_current_hzb`](crate::resolve_current_hzb); those that
/// do not become [`LATE_VISIBLE`](crate::VisibilityStageMask::LATE_VISIBLE) are
/// added to `occluded`.
///
/// Because deferral requires `Reuse` history, a camera cut or history-epoch
/// mismatch leaves both sets empty — nothing is deferred, nothing is culled —
/// matching the conservative guards in [`cull_view`](crate::cull_view). Handles
/// missing from `scene` are skipped. The traversal is linear in `handles` and
/// allocates only the two output sets.
pub fn resolve_two_phase_occlusion(input: TwoPhaseHzbInput<'_>) -> TwoPhaseOcclusion {
    let mut result = TwoPhaseOcclusion::default();
    let history = input.view.history_policy(input.previous_history_epoch);

    for &handle in input.handles {
        let Some(instance) = input.scene.get(handle) else {
            continue;
        };
        let aabb = WorldAabb::from_center_half_extents(
            instance.bounds.center,
            instance.bounds.half_extents,
        );

        let early_occluded = test_bounds_occluded(
            input.view,
            input.previous_pyramid,
            aabb,
            HzbPhase::Previous,
            input.base_query,
        );
        let stage = classify_early_hzb(history, early_occluded);
        if !stage.contains(VisibilityStageMask::LATE_RETEST) {
            continue;
        }

        let pair = (input.view.handle, handle);
        result.late_retest.insert(pair);

        let late_occluded = test_bounds_occluded(
            input.view,
            input.current_pyramid,
            aabb,
            HzbPhase::Current,
            input.base_query,
        );
        if !resolve_current_hzb(stage, late_occluded).contains(VisibilityStageMask::LATE_VISIBLE) {
            result.occluded.insert(pair);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HzbMip, ViewFlags};
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{
        InstanceRecord, SceneBounds, SceneOperation, SceneTransaction,
    };

    /// Reversed-Z clip matrix (column-major): `clip.w = z`, `clip.z = 1`.
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

    fn scene_with(instances: &[(GenerationalHandle, [f32; 3])]) -> CpuRenderScene {
        let mut scene = CpuRenderScene::default();
        let operations = instances
            .iter()
            .map(|&(instance, center)| SceneOperation::Create {
                handle: instance,
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

    /// Near reversed-Z wall (0.9): an occluder close to the camera.
    fn near_wall() -> [f32; 100 * 100] {
        [0.9_f32; 100 * 100]
    }

    /// Far reversed-Z wall (0.05): behind a far instance, so it reveals.
    fn far_wall() -> [f32; 100 * 100] {
        [0.05_f32; 100 * 100]
    }

    fn pyramid(buffer: &[f32]) -> ([HzbMip<'_>; 1], ()) {
        (
            [HzbMip {
                width: 100,
                height: 100,
                texels: buffer,
            }],
            (),
        )
    }

    #[test]
    fn occluded_in_both_phases_is_finally_culled() {
        let prev = near_wall();
        let cur = near_wall();
        let (prev_mips, _) = pyramid(&prev);
        let (cur_mips, _) = pyramid(&cur);
        let previous_pyramid = HzbPyramid::new(&prev_mips);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        let far = handle(1);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0])]);
        let out = resolve_two_phase_occlusion(TwoPhaseHzbInput {
            view: &view(),
            previous_pyramid: &previous_pyramid,
            current_pyramid: &current_pyramid,
            scene: &scene,
            handles: &[far],
            base_query: query(),
            previous_history_epoch: Some(7),
        });
        assert!(out.late_retest.contains(&(view().handle, far)));
        assert!(out.occluded.contains(&(view().handle, far)));
    }

    #[test]
    fn deferred_but_revealed_by_current_hzb_is_not_culled() {
        let prev = near_wall();
        let cur = far_wall();
        let (prev_mips, _) = pyramid(&prev);
        let (cur_mips, _) = pyramid(&cur);
        let previous_pyramid = HzbPyramid::new(&prev_mips);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        let far = handle(1);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0])]);
        let out = resolve_two_phase_occlusion(TwoPhaseHzbInput {
            view: &view(),
            previous_pyramid: &previous_pyramid,
            current_pyramid: &current_pyramid,
            scene: &scene,
            handles: &[far],
            base_query: query(),
            previous_history_epoch: Some(7),
        });
        // Deferred by the early phase, but the current HZB has no occluder in
        // front of it, so it is revealed rather than culled.
        assert!(out.late_retest.contains(&(view().handle, far)));
        assert!(out.occluded.is_empty());
    }

    #[test]
    fn early_visible_instance_is_never_deferred() {
        let prev = near_wall();
        let cur = near_wall();
        let (prev_mips, _) = pyramid(&prev);
        let (cur_mips, _) = pyramid(&cur);
        let previous_pyramid = HzbPyramid::new(&prev_mips);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        // z = 0.5 -> reversed-Z 2.0, in front of the 0.9 wall: visible early.
        let near = handle(1);
        let scene = scene_with(&[(near, [0.0, 0.0, 0.5])]);
        let out = resolve_two_phase_occlusion(TwoPhaseHzbInput {
            view: &view(),
            previous_pyramid: &previous_pyramid,
            current_pyramid: &current_pyramid,
            scene: &scene,
            handles: &[near],
            base_query: query(),
            previous_history_epoch: Some(7),
        });
        assert!(out.late_retest.is_empty());
        assert!(out.occluded.is_empty());
    }

    #[test]
    fn history_reset_defers_and_culls_nothing() {
        let prev = near_wall();
        let cur = near_wall();
        let (prev_mips, _) = pyramid(&prev);
        let (cur_mips, _) = pyramid(&cur);
        let previous_pyramid = HzbPyramid::new(&prev_mips);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        let far = handle(1);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0])]);
        // previous epoch != view.history_epoch -> Reset -> classify keeps the
        // candidate EARLY, so it is never deferred nor culled.
        let out = resolve_two_phase_occlusion(TwoPhaseHzbInput {
            view: &view(),
            previous_pyramid: &previous_pyramid,
            current_pyramid: &current_pyramid,
            scene: &scene,
            handles: &[far],
            base_query: query(),
            previous_history_epoch: Some(6),
        });
        assert!(out.late_retest.is_empty());
        assert!(out.occluded.is_empty());
    }

    #[test]
    fn empty_previous_pyramid_defers_nothing() {
        let empty = HzbPyramid::new(&[]);
        let cur = near_wall();
        let (cur_mips, _) = pyramid(&cur);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        let far = handle(1);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0])]);
        let out = resolve_two_phase_occlusion(TwoPhaseHzbInput {
            view: &view(),
            previous_pyramid: &empty,
            current_pyramid: &current_pyramid,
            scene: &scene,
            handles: &[far],
            base_query: query(),
            previous_history_epoch: Some(7),
        });
        assert!(out.late_retest.is_empty());
        assert!(out.occluded.is_empty());
    }

    #[test]
    fn stale_handles_are_skipped() {
        let prev = near_wall();
        let cur = near_wall();
        let (prev_mips, _) = pyramid(&prev);
        let (cur_mips, _) = pyramid(&cur);
        let previous_pyramid = HzbPyramid::new(&prev_mips);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        let scene = scene_with(&[(handle(1), [0.0, 0.0, 10.0])]);
        let ghost = GenerationalHandle {
            index: 1,
            generation: 99,
        };
        let out = resolve_two_phase_occlusion(TwoPhaseHzbInput {
            view: &view(),
            previous_pyramid: &previous_pyramid,
            current_pyramid: &current_pyramid,
            scene: &scene,
            handles: &[ghost],
            base_query: query(),
            previous_history_epoch: Some(7),
        });
        assert!(out.late_retest.is_empty());
        assert!(out.occluded.is_empty());
    }
}
