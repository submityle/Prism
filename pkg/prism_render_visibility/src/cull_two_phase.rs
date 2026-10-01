//! Two-phase HZB-driven culling entry point.
//!
//! [`cull_view_with_hzb`](crate::cull_view_with_hzb) packages a *single* HZB
//! test (one depth pyramid) into a cull. That is correct only when the pyramid
//! already contains every occluder — which no real frame can guarantee, since
//! the current frame's depth does not exist until something is drawn. Nanite
//! answers this with a two-phase loop over the *previous* and *current* HZBs
//! (see [`resolve_two_phase_occlusion`](crate::resolve_two_phase_occlusion)):
//! an instance is culled only when it is proven hidden in *both* temporal depth
//! hierarchies.
//!
//! [`cull_view_two_phase`] is the production driver for that loop: given both
//! pyramids and the scene slice, it resolves the two-phase occluded set and
//! immediately runs [`cull_view`](crate::cull_view) with it, returning the same
//! `(work, diagnostics)` pair. It reuses [`HzbCullScene`](crate::HzbCullScene)
//! for the scene-side borrows so the single-phase and two-phase drivers share
//! one input shape, and it changes no existing API — `cull_view` and
//! `cull_view_with_hzb` consumers keep working untouched.

use crate::{
    cull_view, resolve_two_phase_occlusion, GpuRenderWorkItem, GpuViewRecord, HzbCullScene,
    HzbPyramid, OcclusionQuery, TwoPhaseHzbInput, VisibilityDiagnostics, VisibilityInput,
    VisibilityStageMask,
};
use alloc::vec::Vec;

/// Resolves the Nanite-style two-phase occlusion set for `view` from the
/// previous and current HZB pyramids, then culls with it.
///
/// The occluded set is computed by
/// [`resolve_two_phase_occlusion`](crate::resolve_two_phase_occlusion) over
/// `inputs.handles`: each instance is tested against `previous_pyramid` in the
/// early phase and, only if deferred, re-tested against `current_pyramid` in
/// the late phase. The resulting
/// [`TwoPhaseOcclusion::occluded`](crate::TwoPhaseOcclusion) set — pairs proven
/// hidden in *both* hierarchies — is borrowed straight into
/// [`VisibilityInput::occluded`](crate::VisibilityInput) with no adapter.
///
/// Because deferral requires `Reuse` history and the per-phase
/// [`HzbTest`](crate::HzbTest) guards are conservative, a camera cut,
/// mismatched history epoch, or empty previous pyramid all collapse to
/// "nothing deferred, nothing culled" — exactly matching the guards in
/// [`cull_view`](crate::cull_view). Compared to
/// [`cull_view_with_hzb`](crate::cull_view_with_hzb) this trades one extra
/// pyramid test per *deferred* candidate for strictly fewer false occlusions:
/// an instance revealed by the current frame's depth is never wrongly culled.
///
/// The extra cost over a bare `cull_view` is one linear pass over
/// `inputs.handles` plus the transient occluded/late-retest sets; no other
/// allocation is introduced.
///
/// # Stage tagging
///
/// `cull_view` stamps every survivor with
/// [`VisibilityStageMask::EARLY`](crate::VisibilityStageMask::EARLY). That is
/// correct for instances that passed the previous-frame HZB, but the two-phase
/// loop also keeps "deferred then revealed" instances — hidden early, revealed
/// by the current HZB. This driver re-tags exactly those survivors (the
/// `late_retest` pairs that were *not* culled) with
/// [`LATE_RETEST`](crate::VisibilityStageMask::LATE_RETEST) |
/// [`LATE_VISIBLE`](crate::VisibilityStageMask::LATE_VISIBLE), matching the
/// verdict in [`resolve_current_hzb`](crate::resolve_current_hzb) so the
/// rasterizer routes them to the late pass instead of the early pass.
pub fn cull_view_two_phase(
    view: &GpuViewRecord,
    previous_pyramid: &HzbPyramid<'_>,
    current_pyramid: &HzbPyramid<'_>,
    base_query: OcclusionQuery,
    inputs: HzbCullScene<'_>,
) -> (Vec<GpuRenderWorkItem>, VisibilityDiagnostics) {
    let occlusion = resolve_two_phase_occlusion(TwoPhaseHzbInput {
        view,
        previous_pyramid,
        current_pyramid,
        scene: inputs.scene,
        handles: inputs.handles,
        base_query,
        previous_history_epoch: inputs.previous_history_epoch,
    });
    let (mut work, diagnostics) = cull_view(
        view,
        VisibilityInput {
            scene: inputs.scene,
            handles: inputs.handles,
            geometry: inputs.geometry,
            materials: inputs.materials,
            previous_lods: inputs.previous_lods,
            occluded: &occlusion.occluded,
            capacity: inputs.capacity,
            previous_history_epoch: inputs.previous_history_epoch,
        },
    );
    // Route the two-phase survivors to the correct raster stage. `cull_view`
    // stamps every item with `EARLY`, which is correct for instances that
    // passed the previous-frame HZB. The remaining `late_retest` pairs are
    // precisely the "deferred then revealed" instances: hidden early, re-tested
    // against the current HZB, and *not* added to `occlusion.occluded` (which
    // `cull_view` already removed). They must be drawn in the late pass, so we
    // tag them `LATE_RETEST | LATE_VISIBLE` — mirroring the single source of
    // truth in `resolve_current_hzb`, where a deferred candidate that survives
    // the current test carries `LATE_RETEST | LATE_VISIBLE`.
    if !occlusion.late_retest.is_empty() {
        for item in &mut work {
            if occlusion.late_retest.contains(&(view.handle, item.scene)) {
                item.visibility_stages =
                    VisibilityStageMask::LATE_RETEST | VisibilityStageMask::LATE_VISIBLE;
            }
        }
    }
    (work, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GeometryLod, GeometryLodChain, HzbMip, ViewFlags, ViewHandle};
    use alloc::collections::BTreeMap;
    use alloc::{vec, vec::Vec};
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{
        CpuRenderScene, GeometryHandle, InstanceRecord, SceneBounds, SceneHandle,
        SceneMaterialHandle, SceneOperation, SceneTransaction,
    };
    use prism_render_material::{Illumination, MaterialDomain, MaterialRecord, MaterialRenderClass};

    fn reverse_z_clip() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    }

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle {
            index,
            generation: 1,
        }
    }

    fn view() -> GpuViewRecord {
        GpuViewRecord {
            handle: handle(3),
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

    fn material_record() -> MaterialRecord {
        MaterialRecord {
            handle: handle(1),
            revision: 1,
            domain: MaterialDomain::Surface,
            render_class: MaterialRenderClass::Opaque,
            illumination: Illumination::Lit,
            features: Default::default(),
            closure_mask: 1,
            surface: Default::default(),
            textures: Vec::new(),
            custom_program: None,
        }
    }

    fn lod_chain() -> GeometryLodChain {
        GeometryLodChain {
            geometry: handle(1),
            lods: vec![GeometryLod {
                level: 0,
                screen_error: 0.01,
                resident: true,
                fallback: true,
            }],
        }
    }

    fn maps() -> (
        BTreeMap<GeometryHandle, GeometryLodChain>,
        BTreeMap<SceneMaterialHandle, MaterialRecord>,
        BTreeMap<(ViewHandle, SceneHandle), u16>,
    ) {
        (
            BTreeMap::from([(handle(1), lod_chain())]),
            BTreeMap::from([(handle(1), material_record())]),
            BTreeMap::new(),
        )
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
                    geometry: handle(1),
                    material: handle(1),
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

    fn mips(buffer: &[f32]) -> [HzbMip<'_>; 1] {
        [HzbMip {
            width: 100,
            height: 100,
            texels: buffer,
        }]
    }

    #[test]
    fn occluded_in_both_phases_is_culled_while_front_survives() {
        let prev = near_wall();
        let cur = near_wall();
        let prev_mips = mips(&prev);
        let cur_mips = mips(&cur);
        let previous_pyramid = HzbPyramid::new(&prev_mips);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        // far at z = 10 -> reversed-Z 0.1 (behind the 0.9 wall in both frames);
        // near at z = 0.5 -> reversed-Z 2.0 (in front, visible early).
        let far = handle(1);
        let near = handle(2);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0]), (near, [0.0, 0.0, 0.5])]);
        let (geometry, materials, previous_lods) = maps();
        let (work, stats) = cull_view_two_phase(
            &view(),
            &previous_pyramid,
            &current_pyramid,
            query(),
            HzbCullScene {
                scene: &scene,
                handles: &[far, near],
                geometry: &geometry,
                materials: &materials,
                previous_lods: &previous_lods,
                capacity: 8,
                previous_history_epoch: Some(7),
            },
        );
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].scene, near);
        assert_eq!(stats.occlusion_rejected, 1);
        assert_eq!(stats.visible_instances, 1);
    }

    #[test]
    fn deferred_but_revealed_by_current_hzb_is_not_culled() {
        // Occluded in the previous frame (near wall) but revealed by the
        // current frame (far wall) -> the two-phase loop must NOT cull it,
        // unlike a single-phase test against the stale previous pyramid.
        let prev = near_wall();
        let cur = far_wall();
        let prev_mips = mips(&prev);
        let cur_mips = mips(&cur);
        let previous_pyramid = HzbPyramid::new(&prev_mips);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        let far = handle(1);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0])]);
        let (geometry, materials, previous_lods) = maps();
        let (work, stats) = cull_view_two_phase(
            &view(),
            &previous_pyramid,
            &current_pyramid,
            query(),
            HzbCullScene {
                scene: &scene,
                handles: &[far],
                geometry: &geometry,
                materials: &materials,
                previous_lods: &previous_lods,
                capacity: 8,
                previous_history_epoch: Some(7),
            },
        );
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].scene, far);
        assert_eq!(stats.occlusion_rejected, 0);
    }

    #[test]
    fn camera_cut_disables_occlusion_so_everything_survives() {
        let prev = near_wall();
        let cur = near_wall();
        let prev_mips = mips(&prev);
        let cur_mips = mips(&cur);
        let previous_pyramid = HzbPyramid::new(&prev_mips);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        let mut cut = view();
        cut.flags = ViewFlags::REVERSE_Z | ViewFlags::CAMERA_CUT;
        let far = handle(1);
        let near = handle(2);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0]), (near, [0.0, 0.0, 0.5])]);
        let (geometry, materials, previous_lods) = maps();
        let (work, stats) = cull_view_two_phase(
            &cut,
            &previous_pyramid,
            &current_pyramid,
            query(),
            HzbCullScene {
                scene: &scene,
                handles: &[far, near],
                geometry: &geometry,
                materials: &materials,
                previous_lods: &previous_lods,
                capacity: 8,
                previous_history_epoch: Some(7),
            },
        );
        assert_eq!(work.len(), 2);
        assert_eq!(stats.occlusion_rejected, 0);
    }

    #[test]
    fn empty_previous_pyramid_culls_nothing_by_occlusion() {
        let cur = near_wall();
        let cur_mips = mips(&cur);
        let previous_pyramid = HzbPyramid::new(&[]);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        let far = handle(1);
        let near = handle(2);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0]), (near, [0.0, 0.0, 0.5])]);
        let (geometry, materials, previous_lods) = maps();
        let (work, stats) = cull_view_two_phase(
            &view(),
            &previous_pyramid,
            &current_pyramid,
            query(),
            HzbCullScene {
                scene: &scene,
                handles: &[far, near],
                geometry: &geometry,
                materials: &materials,
                previous_lods: &previous_lods,
                capacity: 8,
                previous_history_epoch: Some(7),
            },
        );
        // Nothing is deferred in the early phase, so nothing can be culled.
        assert_eq!(work.len(), 2);
        assert_eq!(stats.occlusion_rejected, 0);
    }

    #[test]
    fn deferred_then_revealed_survivor_is_tagged_for_the_late_pass() {
        // Hidden by the previous frame's near wall, revealed by the current
        // frame's far wall: the survivor must carry the late-pass stage so the
        // rasterizer draws it in the second Nanite phase, not the early pass.
        let prev = near_wall();
        let cur = far_wall();
        let prev_mips = mips(&prev);
        let cur_mips = mips(&cur);
        let previous_pyramid = HzbPyramid::new(&prev_mips);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        let far = handle(1);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0])]);
        let (geometry, materials, previous_lods) = maps();
        let (work, _stats) = cull_view_two_phase(
            &view(),
            &previous_pyramid,
            &current_pyramid,
            query(),
            HzbCullScene {
                scene: &scene,
                handles: &[far],
                geometry: &geometry,
                materials: &materials,
                previous_lods: &previous_lods,
                capacity: 8,
                previous_history_epoch: Some(7),
            },
        );
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].scene, far);
        assert!(work[0]
            .visibility_stages
            .contains(VisibilityStageMask::LATE_VISIBLE));
        assert!(work[0]
            .visibility_stages
            .contains(VisibilityStageMask::LATE_RETEST));
        // A deferred-then-revealed item is *not* early visible.
        assert!(!work[0]
            .visibility_stages
            .contains(VisibilityStageMask::EARLY));
    }

    #[test]
    fn early_visible_survivor_keeps_the_early_stage() {
        // The near instance passes the previous-frame HZB outright, so it is an
        // early-pass draw and must retain the EARLY stage even when a different
        // instance in the same cull is deferred to the late pass.
        let prev = near_wall();
        let cur = near_wall();
        let prev_mips = mips(&prev);
        let cur_mips = mips(&cur);
        let previous_pyramid = HzbPyramid::new(&prev_mips);
        let current_pyramid = HzbPyramid::new(&cur_mips);
        let far = handle(1);
        let near = handle(2);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0]), (near, [0.0, 0.0, 0.5])]);
        let (geometry, materials, previous_lods) = maps();
        let (work, _stats) = cull_view_two_phase(
            &view(),
            &previous_pyramid,
            &current_pyramid,
            query(),
            HzbCullScene {
                scene: &scene,
                handles: &[far, near],
                geometry: &geometry,
                materials: &materials,
                previous_lods: &previous_lods,
                capacity: 8,
                previous_history_epoch: Some(7),
            },
        );
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].scene, near);
        assert_eq!(work[0].visibility_stages, VisibilityStageMask::EARLY);
    }
}
