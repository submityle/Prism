//! HZB-driven culling entry point.
//!
//! [`cull_view`](crate::cull_view) is the core culler, but it consumes a
//! pre-computed `occluded` set rather than a depth pyramid: the two-phase HZB
//! projection, gather, and reverse-Z test live in sibling modules and
//! [`resolve_occluded_set`](crate::resolve_occluded_set) turns them into that
//! set. Callers therefore had to run the resolver and thread its result into
//! [`VisibilityInput`](crate::VisibilityInput) by hand, which is easy to get
//! wrong (forgetting the resolve step silently disables occlusion).
//!
//! [`cull_view_with_hzb`] packages the two halves into one call: it resolves
//! the occluded set from the live [`HzbPyramid`](crate::HzbPyramid) and scene,
//! then runs [`cull_view`](crate::cull_view) with that set. This is the real
//! production path for "pyramid in, work items out" — no pre-baked occlusion
//! table — and it changes no existing API, so the existing `cull_view`
//! consumers keep working untouched.

use crate::{
    cull_view, resolve_occluded_set, GeometryLodChain, GpuRenderWorkItem, GpuViewRecord, HzbPhase,
    HzbPyramid, OcclusionQuery, ViewHandle, VisibilityDiagnostics, VisibilityInput,
};
use alloc::{collections::BTreeMap, vec::Vec};
use prism_render_architecture::gpu_scene::{
    CpuRenderScene, GeometryHandle, SceneHandle, SceneMaterialHandle,
};
use prism_render_material::MaterialRecord;

/// Scene-side inputs shared by [`cull_view_with_hzb`] and the
/// [`cull_view`](crate::cull_view) call it drives.
///
/// This mirrors [`VisibilityInput`](crate::VisibilityInput) with the `occluded`
/// field removed: the occluded set is derived from the HZB pyramid inside
/// [`cull_view_with_hzb`] rather than supplied by the caller, so grouping the
/// remaining borrows here keeps the driver to a handful of arguments.
pub struct HzbCullScene<'a> {
    /// The scene whose instances are projected, occlusion-tested, and culled.
    pub scene: &'a CpuRenderScene,
    /// The instance handles to consider, in the order they are visited.
    pub handles: &'a [SceneHandle],
    /// Per-geometry LOD chains used for screen-error LOD selection.
    pub geometry: &'a BTreeMap<GeometryHandle, GeometryLodChain>,
    /// Material records consulted for pass masks and sort keys.
    pub materials: &'a BTreeMap<SceneMaterialHandle, MaterialRecord>,
    /// Previously selected LOD level per `(view, instance)` for hysteresis.
    pub previous_lods: &'a BTreeMap<(ViewHandle, SceneHandle), u16>,
    /// Maximum number of work items to emit before overflow is flagged.
    pub capacity: u32,
    /// The view's history epoch from the previous frame, gating occlusion
    /// reuse exactly as [`cull_view`](crate::cull_view) expects.
    pub previous_history_epoch: Option<u64>,
}

/// Resolves the real HZB occlusion set for `view` and immediately culls with
/// it, returning the same `(work, diagnostics)` pair as
/// [`cull_view`](crate::cull_view).
///
/// The occluded set is computed by
/// [`resolve_occluded_set`](crate::resolve_occluded_set) over the same
/// `inputs.handles`, using `pyramid`, `phase`, and `base_query`; its result is
/// borrowed straight into [`VisibilityInput::occluded`](crate::VisibilityInput)
/// with no adapter. Because the resolver is conservative (only proven-occluded
/// pairs are inserted) and `cull_view` additionally ignores occlusion whenever
/// the view's history is not reused, a camera cut, mismatched history epoch, or
/// empty pyramid all collapse to "nothing culled by occlusion" — matching the
/// guards in [`HzbTest`](crate::HzbTest) and [`cull_view`](crate::cull_view).
///
/// The extra cost over a bare `cull_view` is one linear pass over
/// `inputs.handles` plus the transient occluded set; no other allocation is
/// introduced.
pub fn cull_view_with_hzb(
    view: &GpuViewRecord,
    pyramid: &HzbPyramid<'_>,
    phase: HzbPhase,
    base_query: OcclusionQuery,
    inputs: HzbCullScene<'_>,
) -> (Vec<GpuRenderWorkItem>, VisibilityDiagnostics) {
    let occluded =
        resolve_occluded_set(view, pyramid, inputs.scene, inputs.handles, phase, base_query);
    cull_view(
        view,
        VisibilityInput {
            scene: inputs.scene,
            handles: inputs.handles,
            geometry: inputs.geometry,
            materials: inputs.materials,
            previous_lods: inputs.previous_lods,
            occluded: &occluded,
            capacity: inputs.capacity,
            previous_history_epoch: inputs.previous_history_epoch,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GeometryLod, HzbMip, ViewFlags};
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{
        InstanceRecord, SceneBounds, SceneOperation, SceneTransaction,
    };
    use prism_render_material::{Illumination, MaterialDomain, MaterialRecord, MaterialRenderClass};

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
            // All planes evaluate to `1.0 >= -radius`: always inside.
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

    /// Builds a scene with one small on-screen AABB instance per `(handle,
    /// center)` pair, all sharing geometry/material `handle(1)`.
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

    /// A full-screen single-mip pyramid reading a near reversed-Z depth (0.9):
    /// an occluder close to the camera.
    fn near_wall_buffer() -> [f32; 100 * 100] {
        [0.9_f32; 100 * 100]
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

    #[test]
    fn pyramid_occluded_instance_is_dropped_while_front_instance_survives() {
        let buffer = near_wall_buffer();
        let mips = [HzbMip {
            width: 100,
            height: 100,
            texels: &buffer,
        }];
        let pyramid = HzbPyramid::new(&mips);
        // far at z = 10 -> reversed-Z 0.1 (behind the 0.9 wall);
        // near at z = 0.5 -> reversed-Z 2.0 (in front of the wall).
        let far = handle(1);
        let near = handle(2);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0]), (near, [0.0, 0.0, 0.5])]);
        let (geometry, materials, previous_lods) = maps();
        let (work, stats) = cull_view_with_hzb(
            &view(),
            &pyramid,
            HzbPhase::Current,
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
        // Only the front instance emits work; the far one is occlusion-culled.
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].scene, near);
        assert_eq!(stats.occlusion_rejected, 1);
        assert_eq!(stats.visible_instances, 1);
    }

    #[test]
    fn empty_pyramid_culls_nothing_by_occlusion() {
        let pyramid = HzbPyramid::new(&[]);
        let far = handle(1);
        let near = handle(2);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0]), (near, [0.0, 0.0, 0.5])]);
        let (geometry, materials, previous_lods) = maps();
        let (work, stats) = cull_view_with_hzb(
            &view(),
            &pyramid,
            HzbPhase::Current,
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
    fn camera_cut_disables_occlusion_so_everything_survives() {
        let buffer = near_wall_buffer();
        let mips = [HzbMip {
            width: 100,
            height: 100,
            texels: &buffer,
        }];
        let pyramid = HzbPyramid::new(&mips);
        let mut cut = view();
        cut.flags = ViewFlags::REVERSE_Z | ViewFlags::CAMERA_CUT;
        let far = handle(1);
        let near = handle(2);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0]), (near, [0.0, 0.0, 0.5])]);
        let (geometry, materials, previous_lods) = maps();
        let (work, stats) = cull_view_with_hzb(
            &cut,
            &pyramid,
            HzbPhase::Current,
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
        // Under a camera cut the resolver returns nothing *and* cull_view
        // ignores occlusion, so both instances survive.
        assert_eq!(work.len(), 2);
        assert_eq!(stats.occlusion_rejected, 0);
    }

    #[test]
    fn mismatched_history_epoch_keeps_occluded_instance_visible() {
        let buffer = near_wall_buffer();
        let mips = [HzbMip {
            width: 100,
            height: 100,
            texels: &buffer,
        }];
        let pyramid = HzbPyramid::new(&mips);
        let far = handle(1);
        let scene = scene_with(&[(far, [0.0, 0.0, 10.0])]);
        let (geometry, materials, previous_lods) = maps();
        // previous epoch != view.history_epoch -> history reset -> cull_view
        // disables occlusion even though the resolver would flag `far`.
        let (work, stats) = cull_view_with_hzb(
            &view(),
            &pyramid,
            HzbPhase::Current,
            query(),
            HzbCullScene {
                scene: &scene,
                handles: &[far],
                geometry: &geometry,
                materials: &materials,
                previous_lods: &previous_lods,
                capacity: 8,
                previous_history_epoch: Some(6),
            },
        );
        assert_eq!(work.len(), 1);
        assert_eq!(stats.occlusion_rejected, 0);
    }
}
