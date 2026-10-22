//! Per-view visibility extraction with validated shared frames and surface views.

use alloc::collections::BTreeMap;

use prism_ecs::prelude::{Entity, Resource, World};
use prism_render_architecture::{gpu_scene::SceneHandle, render_fabric::RenderObjectId};
use prism_render_visibility::{GpuRenderWorkItem, GpuViewRecord};

use crate::surface_views::{ExtractedSurfaceViews, SurfaceView};
use crate::visibility::PrismRenderScene;

/// Culling output retained by its complete main-world camera identity.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtractedVisibilityView {
    /// Camera identity, including entity generation.
    pub camera: Entity,
    /// Culling view used to produce this work.
    pub view: GpuViewRecord,
    /// Validated matrices/history consumed by surface encoding.
    pub surface: SurfaceView,
    /// Work visible to this camera only; never unioned with other views.
    pub work: Vec<GpuRenderWorkItem>,
}

/// Per-frame view-scoped culling results and fail-closed extraction rejections.
#[derive(Resource, Default)]
pub struct ExtractedVisibilityViews {
    views: BTreeMap<Entity, ExtractedVisibilityView>,
    rejected: Vec<VisibilityViewRejection>,
}

impl ExtractedVisibilityViews {
    /// Lookup using the complete generational camera entity.
    #[must_use]
    pub fn get(&self, camera: Entity) -> Option<&ExtractedVisibilityView> {
        self.views.get(&camera)
    }

    /// Iterate in deterministic generational entity order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ExtractedVisibilityView> {
        self.views.values()
    }

    /// Cameras rejected by shared-frame, object-identity or surface-view validation.
    #[must_use]
    pub fn rejected(&self) -> &[VisibilityViewRejection] {
        &self.rejected
    }

    /// Number of accepted views.
    #[must_use]
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Whether no valid views were extracted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }
}

/// Why a culling result did not enter the render-world view list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisibilityViewRejectionReason {
    /// Native frame aggregation has not published a shared frame.
    MissingRenderFabricFrame,
    /// Native aggregation reported an error; even an empty result is invalid.
    InvalidRenderFabricFrame,
    /// Shared items were extracted for a different render frame.
    RenderFabricFrameMismatch {
        /// Current render record's frame index (zero when absent).
        expected: u64,
        /// Frame index on the shared item container.
        actual: u64,
    },
    /// Visible work has no matching complete Surface identity in the shared frame.
    MissingSurfaceObject {
        /// Scene identity including generation; index-only matches are invalid.
        handle: SceneHandle,
    },
    /// No camera with the same generational identity passed surface validation.
    MissingValidatedSurfaceView,
    /// Camera matrix differs between visibility culling and surface extraction.
    ClipMatrixMismatch,
    /// Viewport differs between visibility culling and surface extraction.
    ViewportMismatch,
    /// History epoch differs between visibility culling and surface extraction.
    HistoryEpochMismatch,
}

/// A rejected camera and the reason its view-scoped work was discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisibilityViewRejection {
    /// Camera identity, including generation.
    pub camera: Entity,
    /// Fail-closed validation result.
    pub reason: VisibilityViewRejectionReason,
}

/// Extracts per-camera culling after native frame and surface-view extraction.
///
/// A missing, failed or mismatched shared frame rejects all camera results.
/// A missing Surface identity rejects the affected camera's entire result,
/// without retaining partial work or merging it with another camera's list.
pub fn extract_visibility_views(main: &mut World, render: &mut World) {
    let mut result = ExtractedVisibilityViews::default();
    let surfaces = render.get_resource::<ExtractedSurfaceViews>();
    let expected = render
        .get_resource::<crate::RenderRecord>()
        .map_or(0, |record| record.frame_index);
    let frame = match render.get_resource::<crate::ExtractedRenderFabricFrame>() {
        None => Err(VisibilityViewRejectionReason::MissingRenderFabricFrame),
        Some(extracted) if extracted.error().is_some() => {
            Err(VisibilityViewRejectionReason::InvalidRenderFabricFrame)
        }
        Some(extracted) if extracted.frame().frame() != expected => {
            Err(VisibilityViewRejectionReason::RenderFabricFrameMismatch {
                expected,
                actual: extracted.frame().frame(),
            })
        }
        Some(extracted) => Ok(extracted.frame()),
    };
    if let Some(scene) = main.get_resource::<PrismRenderScene>() {
        for (camera, culling) in scene.view_results() {
            let frame = match frame {
                Ok(frame) => frame,
                Err(reason) => {
                    result
                        .rejected
                        .push(VisibilityViewRejection { camera, reason });
                    continue;
                }
            };
            let Some(surface) = surfaces.and_then(|views| views.get(camera)).copied() else {
                result.rejected.push(VisibilityViewRejection {
                    camera,
                    reason: VisibilityViewRejectionReason::MissingValidatedSurfaceView,
                });
                continue;
            };
            let reason = if culling.view.clip_from_world != surface.clip_from_world {
                Some(VisibilityViewRejectionReason::ClipMatrixMismatch)
            } else if culling.view.viewport != surface.viewport {
                Some(VisibilityViewRejectionReason::ViewportMismatch)
            } else if culling.view.history_epoch != surface.history_epoch {
                Some(VisibilityViewRejectionReason::HistoryEpochMismatch)
            } else {
                None
            };
            if let Some(reason) = reason {
                result
                    .rejected
                    .push(VisibilityViewRejection { camera, reason });
                continue;
            }
            if let Some(work) = culling
                .work
                .iter()
                .find(|work| !frame.contains_object(RenderObjectId::Surface(work.scene)))
            {
                result.rejected.push(VisibilityViewRejection {
                    camera,
                    reason: VisibilityViewRejectionReason::MissingSurfaceObject {
                        handle: work.scene,
                    },
                });
                continue;
            }
            result.views.insert(
                camera,
                ExtractedVisibilityView {
                    camera,
                    view: culling.view,
                    surface,
                    work: culling.work.clone(),
                },
            );
        }
    }
    render.insert_resource(result);
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_app::App;
    use prism_math::{Aabb3, Frustum, Mat4, Plane, Vec3};
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{GeometryHandle, SceneMaterialHandle};
    use prism_render_material::fallback_material_record;
    use prism_render_visibility::{GeometryLod, GeometryLodChain};
    use prism_transform::ecs::LocalBounds;
    use prism_transform::{GlobalTransform, Transform};

    use crate::visibility::{
        PrismCamera, PrismFrustum, PrismRenderable, PrismViewVisibility, PrismVisibilityPlugin,
    };

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle::new(index, 1)
    }

    fn test_frustum() -> PrismFrustum {
        PrismFrustum(Frustum::from_planes([
            Plane::new(Vec3::new(1.0, 0.0, 0.0), 10.0),
            Plane::new(Vec3::new(-1.0, 0.0, 0.0), 10.0),
            Plane::new(Vec3::new(0.0, -1.0, 0.0), 10.0),
            Plane::new(Vec3::new(0.0, 1.0, 0.0), 10.0),
            Plane::new(Vec3::new(0.0, 0.0, -1.0), -1.0),
            Plane::new(Vec3::new(0.0, 0.0, 1.0), 100.0),
        ]))
    }

    #[test]
    fn extraction_publishes_empty_result_when_scene_is_absent() {
        let mut main = World::new();
        let mut render = World::new();
        render.insert_resource(ExtractedSurfaceViews::default());

        extract_visibility_views(&mut main, &mut render);

        let extracted = render.resource::<ExtractedVisibilityViews>();
        assert!(extracted.is_empty());
        assert!(extracted.rejected().is_empty());
    }

    fn spawn_instance(app: &mut App, material: SceneMaterialHandle, layer: u32) -> Entity {
        let mut renderable = PrismRenderable::new(handle(1), material);
        renderable.render_layers = layer;
        app.world_mut().spawn((
            renderable,
            LocalBounds(Aabb3::from_center_half_extents(
                Vec3::ZERO,
                Vec3::splat(1.0),
            )),
            GlobalTransform::from_transform(&Transform::from_xyz(0.0, 0.0, -5.0)),
            PrismViewVisibility::default(),
        ))
    }

    fn two_camera_scene() -> (App, Entity, Entity, Entity, Entity) {
        let mut app = App::new();
        app.add_plugins(PrismVisibilityPlugin);
        let geometry: GeometryHandle = handle(1);
        let first_material: SceneMaterialHandle = handle(1);
        let second_material: SceneMaterialHandle = handle(2);
        {
            let scene = app.world_mut().resource_mut::<PrismRenderScene>();
            scene.insert_geometry(GeometryLodChain {
                geometry,
                lods: vec![GeometryLod {
                    level: 0,
                    screen_error: 0.0,
                    resident: true,
                    fallback: true,
                }],
            });
            scene.insert_material(fallback_material_record(first_material, 1));
            scene.insert_material(fallback_material_record(second_material, 1));
        }
        let spawn_camera = |app: &mut App, layer_mask| {
            let mut camera = PrismCamera::new(Mat4::IDENTITY, [0, 0, 640, 480]);
            camera.layer_mask = layer_mask;
            app.world_mut().spawn((
                camera,
                test_frustum(),
                GlobalTransform::from_transform(&Transform::from_xyz(0.0, 0.0, 0.0)),
            ))
        };
        let first = spawn_camera(&mut app, 1);
        let second = spawn_camera(&mut app, 2);
        let first_instance = spawn_instance(&mut app, first_material, 1);
        let second_instance = spawn_instance(&mut app, second_material, 2);

        app.update();
        (app, first, second, first_instance, second_instance)
    }

    fn extract_all(main: &mut World, render: &mut World) {
        crate::fabric_frame::extract_render_fabric_frame(main, render);
        crate::surface_views::extract_surface_views(main, render);
        extract_visibility_views(main, render);
    }

    fn assert_all_rejected(
        render: &World,
        cameras: [Entity; 2],
        reason: VisibilityViewRejectionReason,
    ) {
        let extracted = render.resource::<ExtractedVisibilityViews>();
        assert!(extracted.is_empty());
        assert_eq!(extracted.rejected().len(), cameras.len());
        for camera in cameras {
            assert_eq!(extracted.get(camera), None);
            assert!(extracted
                .rejected()
                .contains(&VisibilityViewRejection { camera, reason }));
        }
    }

    #[test]
    fn two_camera_results_remain_separate_and_generation_keyed() {
        let (mut app, first, second, first_instance, second_instance) = two_camera_scene();
        let scene = app.world().resource::<PrismRenderScene>();
        let first_scene = scene.handle_for(first_instance).unwrap();
        let second_scene = scene.handle_for(second_instance).unwrap();
        let mut render = World::new();
        extract_all(app.world_mut(), &mut render);

        let extracted = render.resource::<ExtractedVisibilityViews>();
        assert_eq!(extracted.len(), 2);
        let first_view = extracted.get(first).expect("first camera was extracted");
        let second_view = extracted.get(second).expect("second camera was extracted");
        assert_eq!(first_view.camera, first);
        assert_eq!(second_view.camera, second);
        assert_ne!(first_view.view.handle, second_view.view.handle);
        assert_eq!(first_view.surface.camera, first);
        assert_eq!(second_view.surface.camera, second);
        assert_eq!(first_view.work.len(), 1);
        assert_eq!(second_view.work.len(), 1);
        assert_eq!(first_view.work[0].scene, first_scene);
        assert_eq!(second_view.work[0].scene, second_scene);
        assert!(extracted.rejected().is_empty());
    }

    #[test]
    fn missing_shared_frame_rejects_both_cameras_and_discards_previous_work() {
        let (mut app, first, second, _, _) = two_camera_scene();
        let mut render = World::new();
        extract_all(app.world_mut(), &mut render);
        assert_eq!(render.resource::<ExtractedVisibilityViews>().len(), 2);
        render.remove_resource::<crate::ExtractedRenderFabricFrame>();

        extract_visibility_views(app.world_mut(), &mut render);

        assert_all_rejected(
            &render,
            [first, second],
            VisibilityViewRejectionReason::MissingRenderFabricFrame,
        );
        extract_all(app.world_mut(), &mut render);
        assert_eq!(render.resource::<ExtractedVisibilityViews>().len(), 2);
        assert!(render
            .resource::<ExtractedVisibilityViews>()
            .rejected()
            .is_empty());
    }

    #[test]
    fn shared_frame_must_match_render_record_even_for_empty_camera_work() {
        let (mut app, first, second, _, _) = two_camera_scene();
        // Only the render-world record is authoritative, including fallback zero.
        app.world_mut().insert_resource(crate::RenderRecord {
            frame_index: 9001,
            ..Default::default()
        });
        app.world_mut()
            .resource_mut::<PrismRenderScene>()
            .set_view_capacity(0);
        app.update();
        let mut render = World::new();
        render.insert_resource(crate::RenderRecord {
            frame_index: 41,
            ..Default::default()
        });
        extract_all(app.world_mut(), &mut render);
        assert_eq!(render.resource::<ExtractedVisibilityViews>().len(), 2);
        assert!(render
            .resource::<ExtractedVisibilityViews>()
            .iter()
            .all(|view| view.work.is_empty()));
        for expected in [42, 40] {
            render.resource_mut::<crate::RenderRecord>().frame_index = expected;
            extract_visibility_views(app.world_mut(), &mut render);
            assert_all_rejected(
                &render,
                [first, second],
                VisibilityViewRejectionReason::RenderFabricFrameMismatch {
                    expected,
                    actual: 41,
                },
            );
        }
        // Without RenderRecord, zero is still a required match, not a bypass.
        render.remove_resource::<crate::RenderRecord>();
        extract_visibility_views(app.world_mut(), &mut render);
        assert_all_rejected(
            &render,
            [first, second],
            VisibilityViewRejectionReason::RenderFabricFrameMismatch {
                expected: 0,
                actual: 41,
            },
        );
        extract_all(app.world_mut(), &mut render);
        assert_eq!(render.resource::<ExtractedVisibilityViews>().len(), 2);
    }

    #[test]
    fn aggregation_error_rejects_views_instead_of_publishing_empty_or_old_frame() {
        let (mut app, first, second, first_instance, _) = two_camera_scene();
        let mut render = World::new();
        extract_all(app.world_mut(), &mut render);
        assert_eq!(render.resource::<ExtractedVisibilityViews>().len(), 2);
        // Corrupt a source bound and synchronize without reculling: the prior
        // valid camera work remains available to exercise the frame-error gate.
        app.world_mut()
            .get_mut::<LocalBounds>(first_instance)
            .unwrap()
            .0 = Aabb3::from_center_half_extents(Vec3::ZERO, Vec3::splat(f32::INFINITY));
        crate::visibility::sync_scene(app.world_mut());
        crate::fabric_frame::extract_render_fabric_frame(app.world_mut(), &mut render);
        assert!(render
            .resource::<crate::ExtractedRenderFabricFrame>()
            .error()
            .is_some());

        extract_visibility_views(app.world_mut(), &mut render);

        assert_all_rejected(
            &render,
            [first, second],
            VisibilityViewRejectionReason::InvalidRenderFabricFrame,
        );
        app.world_mut()
            .get_mut::<LocalBounds>(first_instance)
            .unwrap()
            .0 = Aabb3::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
        app.update();
        extract_all(app.world_mut(), &mut render);
        assert_eq!(render.resource::<ExtractedVisibilityViews>().len(), 2);
        assert!(render
            .resource::<ExtractedVisibilityViews>()
            .rejected()
            .is_empty());
    }

    #[test]
    fn absent_surface_object_rejects_whole_affected_camera_only() {
        let (mut app, first, second, _, _) = two_camera_scene();
        let mut render = World::new();
        extract_all(app.world_mut(), &mut render);
        let added = spawn_instance(&mut app, handle(1), 1);
        app.update();
        let scene = app.world().resource::<PrismRenderScene>();
        let missing = scene.handle_for(added).unwrap();
        assert_eq!(scene.view_result(first).unwrap().work.len(), 2);
        assert!(!render
            .resource::<crate::ExtractedRenderFabricFrame>()
            .frame()
            .contains_object(RenderObjectId::Surface(missing)));
        crate::surface_views::extract_surface_views(app.world_mut(), &mut render);

        extract_visibility_views(app.world_mut(), &mut render);

        let extracted = render.resource::<ExtractedVisibilityViews>();
        assert_eq!(extracted.len(), 1);
        assert!(extracted.get(first).is_none());
        assert_eq!(extracted.get(second).unwrap().work.len(), 1);
        assert_eq!(
            extracted.rejected(),
            &[VisibilityViewRejection {
                camera: first,
                reason: VisibilityViewRejectionReason::MissingSurfaceObject { handle: missing },
            }]
        );
    }

    #[test]
    fn recycled_scene_index_cannot_match_previous_generation() {
        let (mut app, first, second, first_instance, _) = two_camera_scene();
        let old = app
            .world()
            .resource::<PrismRenderScene>()
            .handle_for(first_instance)
            .unwrap();
        let mut render = World::new();
        extract_all(app.world_mut(), &mut render);
        assert!(app.world_mut().despawn(first_instance));
        app.update();
        let replacement = spawn_instance(&mut app, handle(1), 1);
        app.update();
        let current = app
            .world()
            .resource::<PrismRenderScene>()
            .handle_for(replacement)
            .unwrap();
        assert_eq!(old.index, current.index);
        assert_ne!(old.generation, current.generation);
        let frame = render
            .resource::<crate::ExtractedRenderFabricFrame>()
            .frame();
        assert!(frame.contains_object(RenderObjectId::Surface(old)));
        assert!(!frame.contains_object(RenderObjectId::Surface(current)));
        crate::surface_views::extract_surface_views(app.world_mut(), &mut render);

        extract_visibility_views(app.world_mut(), &mut render);

        let extracted = render.resource::<ExtractedVisibilityViews>();
        assert_eq!(extracted.len(), 1);
        assert!(extracted.get(first).is_none());
        assert!(extracted.get(second).is_some());
        assert_eq!(
            extracted.rejected(),
            &[VisibilityViewRejection {
                camera: first,
                reason: VisibilityViewRejectionReason::MissingSurfaceObject { handle: current },
            }]
        );
    }
}
