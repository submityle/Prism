//! Exercise the production visibility-work → mesh queue → draw path, not a
//! synthetic buffer-copy consumer. Hardware checks are opt-in-required via
//! PRISM_REQUIRE_GPU, just like the other renderer hardware tests.

use super::*;
use crate::ecs_columns::{
    EcsColumnUploadError, EcsColumnUploadStats, EcsResidentColumns, GpuColumnSyncReport,
};
use crate::{GpuColumnAppExt, PrismRenderPlugin, RenderApp, RenderAppConfig, RenderFrameStats};
use bytemuck::{Pod, Zeroable};
use prism_app::App;
use prism_render_driver as rhi;
use prism_render_driver_null::NullInstance;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Affine([[f32; 4]; 3]);
impl Component for Affine {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OtherAffine(Affine);
impl Component for OtherAffine {}

struct Extra;
impl Component for Extra {}

fn affine(x: f32) -> Affine {
    Affine([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
}

fn handle(index: u32) -> SceneHandle {
    SceneHandle::new(index, 1)
}

fn entities(app: &mut App) -> (Entity, Entity) {
    // Keep a preceding row in the SAME archetype, to catch accidentally binding
    // offset zero or a packed scene-order row instead of the physical ECS row.
    let first = app
        .world_mut()
        .spawn((affine(-0.75), RenderTransformBinding(handle(9))));
    let visible = app
        .world_mut()
        .spawn((affine(0.75), RenderTransformBinding(handle(3))));
    (first, visible)
}

fn app() -> (App, Entity, Entity) {
    let mut app = App::new();
    app.register_gpu_transform_column::<Affine>().unwrap();
    app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));
    // Idempotent registration works on both sides of plugin installation.
    app.register_gpu_transform_column::<Affine>().unwrap();
    crate::tests::queue_scene_fixture(&mut app);
    let (first, visible) = entities(&mut app);
    (app, first, visible)
}

fn request(world: &World) -> DrawRequest {
    let work = world.resource::<crate::VisibilityWorkList>().items()[0];
    world
        .resource::<crate::SceneRegistry>()
        .resolve_passes(
            &work,
            world.resource::<crate::RenderAssets<crate::RenderMesh>>(),
            Some(crate::RENDER_DEPTH_FORMAT),
            1,
            crate::draw::canonical::MESH,
        )
        .unwrap()
        .remove(0)
}

fn item(request: DrawRequest) -> crate::PhaseItem {
    crate::PhaseItem {
        pipeline: rhi::RenderPipelineId::from_parts(0, 0),
        draw: request.draw,
        geometry: request.geometry,
        vertices: request.vertices,
        indices: request.indices,
        first: request.first,
        count: request.count,
        base_vertex: request.base_vertex,
        indirect: None,
        lod_level: request.lod_level,
        material: request.material,
        scene: request.scene,
        order: request.order,
    }
}

#[derive(Default)]
struct Commands {
    vertices: Vec<(u32, BufferId, u64)>,
    draws: usize,
}
impl crate::MeshRecorder for Commands {
    fn set_pipeline(&mut self, _: rhi::RenderPipelineId) {}
    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferId, offset: u64) {
        self.vertices.push((slot, buffer, offset));
    }
    fn set_index_buffer(&mut self, _: BufferId, _: rhi::IndexFormat, _: u64) {}
    fn draw(&mut self, _: core::ops::Range<u32>, instances: core::ops::Range<u32>) {
        assert_eq!(instances, 0..1);
        self.draws += 1;
    }
    fn draw_indexed(&mut self, _: core::ops::Range<u32>, _: i32, instances: core::ops::Range<u32>) {
        assert_eq!(instances, 0..1);
        self.draws += 1;
    }
    fn draw_indexed_indirect(&mut self, _: prism_render_graph::BufferHandle, _: u64) {
        panic!("unexpected indirect draw");
    }
    fn multi_draw_indirect(&mut self, _: prism_render_graph::BufferHandle, _: u64, _: u32, _: u32) {
        panic!("unexpected indirect draw");
    }
}

fn assert_direct_row(app: &mut App, entity: Entity) -> BufferId {
    let location = app.world().entities().location(entity).unwrap();
    let render = &mut app.sub_app_mut(RenderApp).unwrap().world;
    let column = render
        .resource::<EcsResidentColumns<Affine>>()
        .column(location.archetype_id)
        .unwrap();
    let mut request = request(render);
    let mut transforms = render.remove_resource::<RenderTransforms>().unwrap();
    assert!(transforms
        .decorate(&mut request, render.resource::<MeshPipelines>())
        .unwrap());
    assert_eq!(request.vertices.last(), column.buffer_id().as_ref());
    assert_eq!(request.layout.last().unwrap().array_stride, 48);
    let item = item(request);
    let mut commands = Commands::default();
    render
        .resource::<DrawFunctions>()
        .call(item.draw, &item, &mut commands);
    assert_eq!(
        commands.vertices.last(),
        Some(&(1, column.buffer_id().unwrap(), u64::from(location.row) * 48))
    );
    assert_eq!(commands.draws, 1);
    assert!(
        transforms.buffer().is_none(),
        "no legacy packed GPU transform buffer"
    );
    assert_eq!(transforms.stats().uploaded_bytes, 0);
    render.insert_resource(transforms);
    column.buffer_id().unwrap()
}

#[test]
fn production_draw_binds_resident_physical_row_and_static_frames_upload_nothing() {
    let (mut app, _, visible) = app();
    app.update();
    assert!(app
        .world()
        .resource::<PublishedTransforms>()
        .bytes()
        .is_empty());
    assert!(app
        .world()
        .resource::<PublishedTransforms>()
        .last_error()
        .is_none());
    let buffer = assert_direct_row(&mut app, visible);
    let render = app.get_sub_app(RenderApp).unwrap();
    assert_eq!(
        render.world.resource::<RenderFrameStats>().draws_recorded,
        1
    );
    assert_eq!(
        render
            .world
            .resource::<GpuColumnSyncReport<Affine>>()
            .result
            .unwrap()
            .uploaded_bytes,
        96
    );
    let functions = render.world.resource::<DrawFunctions>().len();
    app.update();
    assert_eq!(assert_direct_row(&mut app, visible), buffer);
    let render = app.get_sub_app(RenderApp).unwrap();
    assert_eq!(
        render
            .world
            .resource::<GpuColumnSyncReport<Affine>>()
            .result
            .unwrap(),
        EcsColumnUploadStats::default()
    );
    assert_eq!(render.world.resource::<DrawFunctions>().len(), functions);
    assert_eq!(
        render.world.resource::<RenderFrameStats>().draws_recorded,
        2
    );
}

#[test]
fn changed_affine_reuses_the_direct_column_then_returns_to_clean_uploads() {
    let (mut app, _, visible) = app();
    app.update();
    let buffer = assert_direct_row(&mut app, visible);
    app.world_mut().increment_change_tick();
    *app.world_mut().get_mut::<Affine>(visible).unwrap() = affine(-0.5);
    app.update();
    assert_eq!(assert_direct_row(&mut app, visible), buffer);
    let report = app
        .get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<GpuColumnSyncReport<Affine>>()
        .result
        .unwrap();
    assert!(report.uploaded_bytes >= 48);
    assert_eq!(report.allocations, 0);
    app.update();
    assert_eq!(assert_direct_row(&mut app, visible), buffer);
    assert_eq!(
        app.get_sub_app(RenderApp)
            .unwrap()
            .world
            .resource::<GpuColumnSyncReport<Affine>>()
            .result
            .unwrap(),
        EcsColumnUploadStats::default()
    );
}

#[test]
fn scene_generation_and_binding_removal_do_not_reuse_a_stale_draw_row() {
    let (mut app, first, visible) = app();
    app.update();
    assert_direct_row(&mut app, visible);
    let render = &app.get_sub_app(RenderApp).unwrap().world;
    let mut stale = request(render);
    let draw = render.resource::<RenderTransforms>().draw.unwrap();
    stale.vertices.push(
        render.resource::<RenderTransforms>().resident.0[&handle(3)]
            .as_ref()
            .unwrap()
            .column
            .buffer_id()
            .unwrap(),
    );
    stale.draw = draw;
    let stale = item(stale);
    *app.world_mut()
        .get_mut::<RenderTransformBinding>(visible)
        .unwrap() = RenderTransformBinding(SceneHandle::new(3, 2));
    app.update();
    let render = &mut app.sub_app_mut(RenderApp).unwrap().world;
    let mut old_request = request(render);
    let mut current_request = request(render);
    current_request.scene = SceneHandle::new(3, 2);
    let mut transforms = render.remove_resource::<RenderTransforms>().unwrap();
    assert!(!transforms
        .decorate(&mut old_request, render.resource::<MeshPipelines>())
        .unwrap());
    assert!(transforms
        .decorate(&mut current_request, render.resource::<MeshPipelines>())
        .unwrap());
    render.insert_resource(transforms);
    let mut commands = Commands::default();
    render
        .resource::<DrawFunctions>()
        .call(draw, &stale, &mut commands);
    assert_eq!(
        commands.draws, 0,
        "old scene generation has no resident row"
    );

    app.world_mut().despawn(first);
    app.world_mut().despawn(visible);
    app.update();
    let render = &app.get_sub_app(RenderApp).unwrap().world;
    assert!(render.resource::<RenderTransforms>().resident.0.is_empty());
    assert_eq!(render.resource::<RenderTransforms>().draw, Some(draw));
    render
        .resource::<DrawFunctions>()
        .call(draw, &stale, &mut commands);
    assert_eq!(commands.draws, 0, "unbound draw closure is cleared");
}

#[test]
fn resident_row_errors_reject_the_draw_and_a_valid_next_extract_recovers() {
    let (mut app, _, visible) = app();
    app.update();
    let buffer = assert_direct_row(&mut app, visible);
    let render = &mut app.sub_app_mut(RenderApp).unwrap().world;
    let rows = render.resource::<ResidentTransformRows>().clone();
    render.resource_mut::<ResidentTransformRows>().0.insert(
        handle(3),
        Err("resident transform upload: DeviceLost".into()),
    );
    prepare_transform_buffer::<NullInstance>(render).unwrap();
    let mut request = request(render);
    let mut transforms = render.remove_resource::<RenderTransforms>().unwrap();
    assert!(transforms
        .decorate(&mut request, render.resource::<MeshPipelines>())
        .unwrap_err()
        .contains("DeviceLost"));
    assert_eq!(transforms.stats().draws_rejected, 1);
    render.insert_resource(transforms);
    render.insert_resource(rows);
    prepare_transform_buffer::<NullInstance>(render).unwrap();
    assert_eq!(assert_direct_row(&mut app, visible), buffer);
}

#[test]
fn failed_preparation_rejects_new_scene_bindings_without_releasing_old_views() {
    let (mut app, _, visible) = app();
    app.update();
    let old_buffer = assert_direct_row(&mut app, visible);
    let render = &mut app.sub_app_mut(RenderApp).unwrap().world;
    let mut next = render.resource::<ResidentTransformRows>().clone();
    let row = next.0.remove(&handle(3)).unwrap();
    let new_handle = SceneHandle::new(3, 2);
    next.0.insert(new_handle, row);
    let mut request = request(render);
    request.scene = new_handle;
    let mut transforms = render.remove_resource::<RenderTransforms>().unwrap();
    transforms.reject_preparation(
        render.resource::<ExtractedTransforms>().0.clone(),
        next,
        "injected retirement wait failure".into(),
    );
    assert_eq!(
        transforms.resident.0[&handle(3)]
            .as_ref()
            .unwrap()
            .column
            .buffer_id(),
        Some(old_buffer),
        "old submitted views must survive until successful retirement"
    );
    assert!(transforms
        .decorate(&mut request, render.resource::<MeshPipelines>())
        .unwrap_err()
        .contains("retirement wait failure"));
    assert_eq!(transforms.stats().draws_rejected, 1);
    render.insert_resource(transforms);
    prepare_transform_buffer::<NullInstance>(render).unwrap();
    assert!(render
        .resource::<RenderTransforms>()
        .pending_resident
        .0
        .is_empty());
    assert_eq!(assert_direct_row(&mut app, visible), old_buffer);
}

#[test]
fn resident_retirement_only_drains_replaced_allocations_on_the_same_device() {
    let (mut app, first, visible) = app();
    app.update();
    let render = &app.get_sub_app(RenderApp).unwrap().world;
    let old = render.resource::<ResidentTransformRows>().clone();
    let device: Arc<dyn RenderDevice + Send + Sync> = render
        .resource::<RenderRuntimeResource<NullInstance>>()
        .runtime()
        .device_handle();
    assert!(!old.releases_on_device(&old, &device));
    let mut shared = old.clone();
    shared.0.remove(&handle(9));
    assert!(
        !old.releases_on_device(&shared, &device),
        "another row keeps the same allocation reachable"
    );
    assert!(old.releases_on_device(&ResidentTransformRows::default(), &device));

    app.world_mut().insert(first, Extra);
    app.world_mut().insert(visible, Extra);
    app.update();
    let render = &app.get_sub_app(RenderApp).unwrap().world;
    let new = render.resource::<ResidentTransformRows>();
    assert!(old.releases_on_device(new, &device));
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&rhi::RequestAdapterOptions::default())
        .unwrap();
    let (other_device, _) = instance
        .create_device(&adapter, &rhi::DeviceDescriptor::default())
        .unwrap();
    let other_device: Arc<dyn RenderDevice + Send + Sync> = Arc::new(other_device);
    assert!(
        !old.releases_on_device(new, &other_device),
        "lost/replaced devices do not wait on the replacement queue"
    );
}

#[test]
fn resident_draw_remaps_swap_remove_migration_and_snapshot_restore() {
    let (mut app, first, visible) = app();
    app.world_mut().register_snapshot_component::<Affine>();
    app.world_mut()
        .register_snapshot_component::<RenderTransformBinding>();
    let snapshot = app.world().snapshot();
    app.update();
    let initial = assert_direct_row(&mut app, visible);
    app.world_mut().despawn(first);
    app.update();
    assert_eq!(app.world().entities().location(visible).unwrap().row, 0);
    assert_eq!(assert_direct_row(&mut app, visible), initial);
    app.world_mut().insert(visible, Extra);
    app.update();
    let migrated = assert_direct_row(&mut app, visible);
    assert_ne!(migrated, initial);
    app.world_mut().restore(&snapshot);
    app.update();
    let restored = assert_direct_row(&mut app, visible);
    assert_ne!(restored, migrated);
    assert_eq!(app.world().entities().location(visible).unwrap().row, 1);
    assert_eq!(
        app.get_sub_app(RenderApp)
            .unwrap()
            .world
            .resource::<GpuColumnSyncReport<Affine>>()
            .result
            .unwrap()
            .uploaded_bytes,
        96
    );
}

#[test]
fn duplicate_sources_and_missing_column_fail_closed_and_recover() {
    let (mut app, _, visible) = app();
    app.register_gpu_transform_column::<OtherAffine>().unwrap();
    app.world_mut().insert(visible, OtherAffine(affine(-0.75)));
    app.update();
    let render = app.get_sub_app(RenderApp).unwrap();
    assert_eq!(
        render.world.resource::<RenderFrameStats>().draws_recorded,
        0
    );
    assert!(render
        .world
        .resource::<RenderTransforms>()
        .last_error()
        .unwrap()
        .contains("duplicate resident"));
    app.world_mut().remove::<OtherAffine>(visible);
    app.update();
    assert_direct_row(&mut app, visible);
    assert_eq!(
        app.get_sub_app(RenderApp)
            .unwrap()
            .world
            .resource::<RenderFrameStats>()
            .draws_recorded,
        1
    );
    app.world_mut().remove::<Affine>(visible);
    app.update();
    let render = app.get_sub_app(RenderApp).unwrap();
    assert_eq!(
        render.world.resource::<RenderFrameStats>().draws_recorded,
        1
    );
    assert!(render
        .world
        .resource::<RenderTransforms>()
        .last_error()
        .unwrap()
        .contains("missing global"));
    app.world_mut().insert(visible, affine(0.25));
    app.update();
    assert_direct_row(&mut app, visible);
}

#[test]
fn legacy_and_resident_sources_coexist_but_duplicate_scene_bindings_are_rejected() {
    let (mut app, _, visible) = app();
    let legacy = app.world_mut().spawn((
        prism_transform::Transform::IDENTITY,
        RenderTransformBinding(handle(20)),
    ));
    app.update();
    assert_eq!(
        app.world().resource::<PublishedTransforms>().bytes().len(),
        48
    );
    let render = app.get_sub_app(RenderApp).unwrap();
    assert!(render
        .world
        .resource::<RenderTransforms>()
        .buffer()
        .is_some());
    assert_eq!(
        render.world.resource::<RenderFrameStats>().draws_recorded,
        1
    );
    *app.world_mut()
        .get_mut::<RenderTransformBinding>(legacy)
        .unwrap() = RenderTransformBinding(handle(3));
    app.update();
    let render = app.get_sub_app(RenderApp).unwrap();
    assert_eq!(
        render.world.resource::<RenderFrameStats>().draws_recorded,
        1
    );
    assert!(render
        .world
        .resource::<RenderTransforms>()
        .last_error()
        .unwrap()
        .contains("duplicate"));
    app.world_mut().despawn(legacy);
    app.update();
    assert_direct_row(&mut app, visible);
}

#[test]
fn generic_column_registration_does_not_opt_into_mesh_transform_semantics() {
    let mut app = App::new();
    app.register_gpu_column::<Affine>().unwrap();
    app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));
    crate::tests::queue_scene_fixture(&mut app);
    app.world_mut().spawn((
        affine(0.75),
        prism_transform::Transform::IDENTITY,
        RenderTransformBinding(handle(3)),
    ));
    app.update();
    let render = app.get_sub_app(RenderApp).unwrap();
    assert!(render
        .world
        .resource::<RenderTransforms>()
        .resident
        .0
        .is_empty());
    assert_eq!(
        render
            .world
            .resource::<RenderTransforms>()
            .stats()
            .uploaded_bytes,
        48
    );
    assert_eq!(
        render.world.resource::<RenderFrameStats>().draws_recorded,
        1
    );
}

#[test]
fn invalid_affine_layout_is_rejected_without_registering_a_column() {
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Short([f32; 4]);
    impl Component for Short {}
    let mut app = App::new();
    assert!(matches!(
        app.register_gpu_transform_column::<Short>(),
        Err(EcsColumnUploadError::InvalidTransformStride { stride: 16 })
    ));
    assert!(app.world().components().id_of::<Short>().is_none());
}

#[test]
fn hardware_resident_column_changes_pixels_and_clean_frames_keep_draws_without_uploads() {
    use prism_render_driver_wgpu::WgpuInstance;
    const SIZE: u32 = 16;
    let extent = rhi::Extent3d::new_2d(SIZE, SIZE);
    if prism_render_runtime::RenderRuntime::<WgpuInstance>::wgpu(extent).is_err() {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "required GPU is unavailable"
        );
        eprintln!("SKIPPED resident-column production draw: no adapter");
        return;
    }
    fn pixels(app: &App) -> Vec<u8> {
        let world = &app.get_sub_app(RenderApp).unwrap().world;
        let runtime = world
            .resource::<RenderRuntimeResource<WgpuInstance>>()
            .runtime();
        assert!(runtime.wait_idle());
        assert_eq!(world.resource::<RenderFrameStats>().frames_failed, 0);
        runtime
            .queue()
            .read_texture_rgba8(
            world.resource::<crate::RenderTarget>().texture(),
            SIZE,
            SIZE,
            )
            .expect("resident transform output readback must complete")
    }
    fn pixel(bytes: &[u8], x: usize) -> &[u8] {
        let start = (8 * SIZE as usize + x) * 4;
        &bytes[start..start + 4]
    }
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        WgpuInstance::new(),
        RenderAppConfig::new(extent),
    ));
    app.register_gpu_transform_column::<Affine>().unwrap();
    crate::tests::queue_scene_fixture(&mut app);
    let (first, visible) = entities(&mut app);
    app.update();
    let right = pixels(&app);
    assert_eq!(pixel(&right, 13), &[255, 255, 255, 255]);
    assert_eq!(pixel(&right, 8), &[0, 0, 0, 255]);
    let buffer = assert_direct_row(&mut app, visible);

    app.update();
    assert_eq!(pixels(&app), right);
    assert_eq!(assert_direct_row(&mut app, visible), buffer);
    assert_eq!(
        app.get_sub_app(RenderApp)
            .unwrap()
            .world
            .resource::<GpuColumnSyncReport<Affine>>()
            .result
            .unwrap(),
        EcsColumnUploadStats::default()
    );

    app.world_mut().increment_change_tick();
    *app.world_mut().get_mut::<Affine>(visible).unwrap() = affine(-0.75);
    app.update();
    let left = pixels(&app);
    assert_eq!(pixel(&left, 2), &[255, 255, 255, 255]);
    assert_eq!(pixel(&left, 13), &[0, 0, 0, 255]);
    assert_ne!(left, right);
    assert_eq!(assert_direct_row(&mut app, visible), buffer);
    let changed = app
        .get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<GpuColumnSyncReport<Affine>>()
        .result
        .unwrap();
    assert!(changed.uploaded_bytes >= 48);
    assert_eq!(changed.allocations, 0);

    app.world_mut().despawn(first);
    app.update();
    assert_eq!(
        pixels(&app),
        left,
        "swap removal must update the draw's physical row"
    );
    assert_direct_row(&mut app, visible);
    app.world_mut().insert(visible, Extra);
    app.update();
    assert_eq!(
        pixels(&app),
        left,
        "archetype migration must rebind the new column"
    );
    assert_ne!(assert_direct_row(&mut app, visible), buffer);
    app.update();
    assert_eq!(pixels(&app), left);
    let render = app.get_sub_app(RenderApp).unwrap();
    assert_eq!(
        render
            .world
            .resource::<GpuColumnSyncReport<Affine>>()
            .result
            .unwrap(),
        EcsColumnUploadStats::default()
    );
    assert_eq!(
        render.world.resource::<RenderFrameStats>().draws_recorded,
        6
    );
    assert!(render
        .world
        .resource::<ExtractedTransforms>()
        .bytes()
        .is_empty());
}
