//! Regressions for material publication reaching Queue in the same Render run.
//! These drive the plugin's real schedules; no test calls the upload system.
//! Assertions cover the CPU ABI mirror and real queued bindings, not shader
//! decoding or the contents of GPU memory on the null backend.

use alloc::vec;

use prism_app::App;
use prism_ecs::{
    prelude::{Res, ResMut, Resource},
    schedule::{IntoSystemConfigs, Schedules},
};
use prism_render_architecture::{abi::GenerationalHandle, gpu_scene::GpuCompletionValue};
use prism_render_driver::Extent3d;
use prism_render_driver_null::NullInstance;
use prism_render_material::{
    fallback_material_record, inactive_material_header, DynamicChannels, MaterialRecord,
};
use prism_render_visibility::GpuRenderWorkItem;

use super::{MaterialParameterBuffers, RenderMaterials};
use crate::surface_view_frame::SurfaceViewQueues;
use crate::{
    label::{Render, RenderApp},
    mesh::RenderMesh,
    queue::RenderQueues,
    render_asset::RenderAssets,
    scene::{SceneRegistry, VisibilityWorkList},
    sets::{RenderPhase, RenderSystems},
    PrismRenderPlugin, RenderAppConfig,
};

#[derive(Resource)]
struct QueueProbe {
    handle: GenerationalHandle,
    revision: Option<u32>,
    dynamic: DynamicChannels,
    header_checks: usize,
    draw_checks: usize,
}

fn assert_header_at_queue(
    buffers: Res<MaterialParameterBuffers>,
    materials: Res<RenderMaterials>,
    scene: Res<SceneRegistry>,
    mut probe: ResMut<QueueProbe>,
) {
    let row = buffers
        .header(probe.handle.index)
        .expect("publication must reach the header mirror before Queue");
    if let Some(revision) = probe.revision {
        let registry = materials.registry();
        let record = registry
            .get(probe.handle)
            .expect("the live registry record");
        assert_eq!(record.revision, revision);
        assert_eq!(scene.material(probe.handle), Some(record));
        // These fixtures never install an execution binding. Do not derive the
        // expected inert tail from the state whose reset we are checking.
        assert_eq!(registry.execution_binding_at(probe.handle.index), None);
        assert_eq!(
            registry.dynamic_channels_at(probe.handle.index),
            probe.dynamic,
            "publication/reuse must reset runtime channels, independently of the upload"
        );
        let mut expected = record.header_with_execution(
            row.parameter_offset,
            row.texture_offset,
            registry.snapshot().epoch,
            None,
        );
        let [wetness, snow, damage, dissolve] = probe.dynamic.to_words();
        expected.dynamic_wetness = wetness;
        expected.dynamic_snow = snow;
        expected.dynamic_damage = damage;
        expected.dynamic_dissolve = dissolve;
        assert_eq!(
            *row, expected,
            "Queue must see this frame's generation, revision and complete runtime tail"
        );
        assert_eq!(
            buffers.parameters_of(probe.handle.index),
            Some(record.packed_parameters().pack().as_slice()),
            "Queue must see this frame's authored parameters too"
        );
    } else {
        assert!(materials.registry().get(probe.handle).is_none());
        assert_eq!(
            *row,
            inactive_material_header(probe.handle.generation),
            "retirement must clear the complete header before Queue"
        );
        assert!(buffers.parameters_of(probe.handle.index).is_none());
        assert!(buffers.textures_of(probe.handle.index).is_none());
    }
    probe.header_checks += 1;
}

// QueueMeshes follows all Queue systems, including the production fill_queues,
// but precedes Prepare and Cleanup. Inspect the queue before it is cleared.
fn assert_draw_after_queue(
    camera_queues: Res<SurfaceViewQueues>,
    scene: Res<SceneRegistry>,
    meshes: Res<RenderAssets<RenderMesh>>,
    work: Res<VisibilityWorkList>,
    mut probe: ResMut<QueueProbe>,
) {
    assert_eq!(probe.header_checks, probe.draw_checks + 1);
    assert!(!work.is_empty(), "the real queue must receive work");
    for item in work.items() {
        // Even retired/stale items must pass scene/geometry resolution. Their
        // absence from the queue must not merely mean the fixture is broken.
        let requests = scene
            .resolve_passes(
                item,
                &meshes,
                Some(crate::frame::RENDER_DEPTH_FORMAT),
                1,
                crate::draw::canonical::MESH,
            )
            .expect("live and stale work must resolve before the ABI header gate");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].material, item.material);
        assert_eq!(requests[0].draw, crate::draw::canonical::MESH);
    }
    let queues = camera_queues
        .iter()
        .next()
        .expect("the admitted camera queue")
        .queues
        .clone();
    let expected_count = usize::from(probe.revision.is_some());
    assert_eq!(
        queues.len(),
        expected_count,
        "the live material must queue immediately; retired/stale generations must not queue"
    );
    if probe.revision.is_some() {
        let draws = queues.phase(RenderPhase::Opaque);
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].material, probe.handle);
        assert_eq!(draws[0].draw, crate::draw::canonical::MESH);
        let groups = queues.dynamic_bind_groups(RenderPhase::Opaque, 0);
        assert!(
            groups.iter().any(|group| {
                group.identity == crate::draw::canonical::SURFACE_MATERIAL
                    && group.index == crate::draw::canonical::SURFACE_GROUP
                    && !group.buffers.is_empty()
            }),
            "exercise the default mesh's actual material binding, not an unbound draw"
        );
    }
    probe.draw_checks += 1;
}

struct Fixture {
    app: App,
    work: GpuRenderWorkItem,
    frames: usize,
}

impl Fixture {
    fn new() -> Self {
        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::with_config(
            NullInstance::new(),
            RenderAppConfig::new(Extent3d::new_2d(8, 8)).with_material_capacity(8),
        ));
        // Shared native mesh/camera fixture publishes matching scene and ABI
        // records, but never runs a frame or synchronizes the material buffers.
        crate::tests::queue_scene_fixture(&mut app);
        let render = app
            .sub_app_mut(RenderApp)
            .expect("the plugin render sub-app");
        let work = render.world.resource::<VisibilityWorkList>().items()[0];
        render.world.insert_resource(QueueProbe {
            handle: work.material,
            revision: Some(1),
            dynamic: DynamicChannels::NONE,
            header_checks: 0,
            draw_checks: 0,
        });
        let schedule = render
            .world
            .resource_mut::<Schedules>()
            .get_mut(Render)
            .expect("the plugin's real Render schedule");
        schedule.add_systems(assert_header_at_queue.in_set(RenderSystems::Queue));
        schedule.add_systems(assert_draw_after_queue.in_set(RenderSystems::QueueMeshes));
        Self {
            app,
            work,
            frames: 0,
        }
    }

    fn publish(&mut self, record: MaterialRecord) {
        let instance = self
            .app
            .world()
            .resource::<crate::tests::QueueSceneFixtureIdentity>()
            .instance;
        self.app
            .world_mut()
            .get_mut::<crate::visibility::PrismRenderable>(instance)
            .expect("fixture renderable remains alive")
            .material = record.handle;
        self.app
            .world_mut()
            .resource_mut::<crate::visibility::PrismRenderScene>()
            .insert_material(record.clone());
        let render = self.app.sub_app_mut(RenderApp).unwrap();
        render
            .world
            .resource_mut::<RenderMaterials>()
            .publish(record.clone())
            .expect("publish the next material revision");
        render
            .world
            .resource_mut::<SceneRegistry>()
            .insert_material(record);
        render.world.resource_mut::<QueueProbe>().dynamic = DynamicChannels::NONE;
    }

    fn set_dynamic(&mut self, channels: DynamicChannels) {
        let render = self.app.sub_app_mut(RenderApp).unwrap();
        render
            .world
            .resource_mut::<RenderMaterials>()
            .registry_mut()
            .set_dynamic_channels(self.work.material, channels)
            .unwrap();
        render.world.resource_mut::<QueueProbe>().dynamic =
            DynamicChannels::from_words(channels.to_words());
    }

    fn retire(&mut self) {
        self.app
            .sub_app_mut(RenderApp)
            .unwrap()
            .world
            .resource_mut::<RenderMaterials>()
            .registry_mut()
            .retire(self.work.material, GpuCompletionValue(0))
            .unwrap();
    }

    fn reuse(&mut self) -> GenerationalHandle {
        let registry = self
            .app
            .sub_app_mut(RenderApp)
            .unwrap()
            .world
            .resource_mut::<RenderMaterials>()
            .registry_mut();
        assert_eq!(registry.reclaim_completed(GpuCompletionValue(0)), 1);
        let handle = registry.allocate().unwrap();
        assert_eq!(handle.index, self.work.material.index);
        assert_ne!(handle.generation, self.work.material.generation);
        handle
    }

    fn frame(&mut self, revision: Option<u32>, stale: Option<GenerationalHandle>) {
        let render = self.app.sub_app_mut(RenderApp).unwrap();
        let probe = render.world.resource_mut::<QueueProbe>();
        probe.handle = self.work.material;
        probe.revision = revision;
        let mut work = vec![self.work];
        if let Some(material) = stale {
            work.push(GpuRenderWorkItem {
                material,
                ..self.work
            });
        }
        render.world.resource_mut::<VisibilityWorkList>().set(work);
        self.app.update();
        self.frames += 1;
        let render = self.app.get_sub_app(RenderApp).unwrap();
        let probe = render.world.resource::<QueueProbe>();
        assert_eq!(
            probe.header_checks, self.frames,
            "Queue probe ran every frame"
        );
        assert_eq!(probe.draw_checks, self.frames, "draw probe ran every frame");
    }
}

#[test]
fn first_publication_is_bound_in_the_same_render_queue() {
    let mut f = Fixture::new();
    // No warm-up frame: this also exercises the initial empty runtime tail.
    f.frame(Some(1), None);
}

#[test]
fn consecutive_revisions_never_lag_behind_the_render_queue() {
    let mut f = Fixture::new();
    f.frame(Some(1), None);
    for revision in 2_u32..=5 {
        let mut record = fallback_material_record(f.work.material, u64::from(revision));
        record.surface.base_color = [revision as f32 / 8.0, 0.25, 0.5, 1.0];
        f.publish(record);
        // Alternate set and clear without a catch-up frame between revisions.
        if revision % 2 == 0 {
            f.set_dynamic(DynamicChannels {
                wetness: 0.5,
                ..DynamicChannels::NONE
            });
        }
        f.frame(Some(revision), None);
    }
}

#[test]
fn runtime_only_changes_and_clear_reach_queue_without_a_new_revision() {
    let mut f = Fixture::new();
    f.frame(Some(1), None);
    f.set_dynamic(DynamicChannels {
        wetness: 0.25,
        snow: 0.5,
        damage: 0.75,
        dissolve: 0.625,
    });
    f.frame(Some(1), None);
    f.set_dynamic(DynamicChannels::NONE);
    f.frame(Some(1), None);
    // The unchanged frame must preserve the explicit clear, not restore an
    // older tail from a revision-keyed cache.
    f.frame(Some(1), None);
}

#[test]
fn retirement_and_later_reuse_are_visible_before_each_render_queue() {
    let mut f = Fixture::new();
    f.set_dynamic(DynamicChannels {
        damage: 0.75,
        ..DynamicChannels::NONE
    });
    f.frame(Some(1), None);
    let stale = f.work.material;
    f.retire();
    // Keep the stale scene record/work: the header must reject it this frame.
    f.frame(None, None);
    let handle = f.reuse();
    f.publish(fallback_material_record(handle, 1));
    f.work.material = handle;
    f.frame(Some(1), Some(stale));
}

#[test]
fn coalesced_retire_reuse_replaces_generation_in_the_same_render_queue() {
    let mut f = Fixture::new();
    f.set_dynamic(DynamicChannels {
        dissolve: 0.625,
        ..DynamicChannels::NONE
    });
    f.frame(Some(1), None);
    let stale = f.work.material;
    f.retire();
    let handle = f.reuse();
    // Deliberately retain the same revision: generation alone distinguishes the
    // new row. There is no intervening frame to upload the inactive old row.
    f.publish(fallback_material_record(handle, 1));
    f.work.material = handle;
    f.frame(Some(1), Some(stale));
}
