use super::*;
use crate::{PrismRenderPlugin, RenderApp, RenderAppConfig, RenderFrameStats};
use prism_app::App;
use prism_ecs::hierarchy::ChildOf;
use prism_render_driver as rhi;
use prism_render_driver_null::{NullInstance, NullQueue};
use prism_transform::Transform;
use std::sync::Mutex;

#[derive(Default)]
struct Writes(Mutex<Vec<(rhi::BufferId, u64, Vec<u8>)>>);
impl RenderQueue for Writes {
    fn write_buffer(&self, buffer: rhi::BufferId, offset: u64, data: &[u8]) {
        self.0.lock().unwrap().push((buffer, offset, data.to_vec()));
    }
    fn write_texture(
        &self,
        destination: rhi::TextureWrite,
        data: &[u8],
        layout: rhi::ImageDataLayout,
    ) {
        NullQueue.write_texture(destination, data, layout);
    }
    fn submit(&self, commands: &[rhi::CommandBuffer]) {
        NullQueue.submit(commands);
    }
}

fn device() -> Arc<dyn RenderDevice + Send + Sync> {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&rhi::RequestAdapterOptions::default())
        .unwrap();
    let (device, _) = instance
        .create_device(&adapter, &rhi::DeviceDescriptor::default())
        .unwrap();
    Arc::new(device)
}

fn translation(bytes: &[u8], row: usize) -> [f32; 3] {
    core::array::from_fn(|axis| {
        let offset = row * STRIDE + axis * 16 + 12;
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    })
}

fn scene(index: u32, generation: u32) -> SceneHandle {
    SceneHandle::new(index, generation)
}

fn spawn(world: &mut World, handle: SceneHandle, x: f32) -> Entity {
    world.spawn((
        Transform::from_xyz(x, 0.0, 0.0),
        RenderTransformBinding(handle),
    ))
}

#[test]
fn hierarchy_publication_and_extraction_are_complete_and_immutable() {
    let mut main = World::new();
    let mut render = World::new();
    let parent = main.spawn(Transform::from_xyz(2.0, 0.0, 0.0));
    let child = spawn(&mut main, scene(3, 1), 4.0);
    main.register_child_of();
    main.add_relation::<ChildOf>(child, parent);
    publish_transforms(&mut main);
    extract_transforms(&mut main, &mut render);
    assert_eq!(
        translation(render.resource::<ExtractedTransforms>().bytes(), 0),
        [6.0, 0.0, 0.0]
    );
    let held = render.resource::<ExtractedTransforms>().0.clone();

    main.increment_change_tick();
    *main.get_mut::<Transform>(parent).unwrap() = Transform::from_xyz(8.0, 0.0, 0.0);
    publish_transforms(&mut main);
    assert_eq!(
        translation(main.resource::<PublishedTransforms>().bytes(), 0),
        [12.0, 0.0, 0.0]
    );
    // A later publish/rotation cannot mutate an extracted or held snapshot.
    publish_transforms(&mut main);
    publish_transforms(&mut main);
    assert_eq!(translation(&held.bytes, 0), [6.0, 0.0, 0.0]);
    assert_eq!(render.resource::<ExtractedTransforms>().version(), 1);
    extract_transforms(&mut main, &mut render);
    assert_eq!(
        translation(render.resource::<ExtractedTransforms>().bytes(), 0),
        [12.0, 0.0, 0.0]
    );
}

#[test]
fn skipped_extracts_upload_all_changes_since_last_consumption_and_static_frames_write_nothing() {
    let mut main = World::new();
    let a = spawn(&mut main, scene(1, 1), 1.0);
    let b = spawn(&mut main, scene(2, 1), 2.0);
    publish_transforms(&mut main);
    let mut gpu = RenderTransforms::default();
    let device = device();
    let writes = Writes::default();
    gpu.sync(
        main.resource::<PublishedTransforms>().snapshot.clone(),
        device.clone(),
        &writes,
    )
    .unwrap();
    assert_eq!(gpu.stats.uploaded_bytes, 96);

    main.increment_change_tick();
    *main.get_mut::<Transform>(a).unwrap() = Transform::from_xyz(10.0, 0.0, 0.0);
    publish_transforms(&mut main); // Render misses this frame's dirty set.
    main.increment_change_tick();
    *main.get_mut::<Transform>(b).unwrap() = Transform::from_xyz(20.0, 0.0, 0.0);
    publish_transforms(&mut main);
    writes.0.lock().unwrap().clear();
    gpu.sync(
        main.resource::<PublishedTransforms>().snapshot.clone(),
        device.clone(),
        &writes,
    )
    .unwrap();
    assert_eq!(gpu.stats.upload_calls, 1);
    let captured = writes.0.lock().unwrap();
    assert_eq!(translation(&captured[0].2, 0)[0], 10.0);
    assert_eq!(translation(&captured[0].2, 1)[0], 20.0);
    drop(captured);
    publish_transforms(&mut main);
    gpu.sync(
        main.resource::<PublishedTransforms>().snapshot.clone(),
        device,
        &writes,
    )
    .unwrap();
    assert_eq!(gpu.stats, TransformUploadStats::default());
}

#[test]
fn dirty_upload_uses_row_offsets_and_device_replacement_replays_clean_bytes() {
    let mut main = World::new();
    spawn(&mut main, scene(1, 1), 1.0);
    let b = spawn(&mut main, scene(2, 1), 2.0);
    publish_transforms(&mut main);
    let first_device = device();
    let writes = Writes::default();
    let mut gpu = RenderTransforms::default();
    gpu.sync(
        main.resource::<PublishedTransforms>().snapshot.clone(),
        first_device.clone(),
        &writes,
    )
    .unwrap();
    main.increment_change_tick();
    *main.get_mut::<Transform>(b).unwrap() = Transform::from_xyz(9.0, 0.0, 0.0);
    publish_transforms(&mut main);
    writes.0.lock().unwrap().clear();
    gpu.sync(
        main.resource::<PublishedTransforms>().snapshot.clone(),
        first_device,
        &writes,
    )
    .unwrap();
    assert_eq!(gpu.stats.uploaded_bytes, 48);
    assert_eq!(writes.0.lock().unwrap()[0].1, 48);

    gpu.sync(
        main.resource::<PublishedTransforms>().snapshot.clone(),
        device(),
        &writes,
    )
    .unwrap();
    assert_eq!(gpu.stats.allocations, 1);
    assert_eq!(
        gpu.stats.uploaded_bytes, 96,
        "a clean CPU publication must replay on a new device"
    );
}

#[test]
fn removal_compacts_rows_and_scene_slot_reuse_does_not_alias_old_generation() {
    let mut main = World::new();
    let a = spawn(&mut main, scene(1, 1), 1.0);
    spawn(&mut main, scene(2, 1), 2.0);
    publish_transforms(&mut main);
    main.despawn(a);
    let replacement = spawn(&mut main, scene(1, 2), 7.0);
    publish_transforms(&mut main);
    let snapshot = &main.resource::<PublishedTransforms>().snapshot;
    assert!(!snapshot.rows.contains_key(&scene(1, 1)));
    assert_eq!(
        translation(&snapshot.bytes, snapshot.rows[&scene(1, 2)])[0],
        7.0
    );
    main.despawn(replacement);
    publish_transforms(&mut main);
    let snapshot = &main.resource::<PublishedTransforms>().snapshot;
    assert_eq!(snapshot.rows[&scene(2, 1)], 0);
    assert_eq!(snapshot.bytes.len(), STRIDE);
    assert_eq!(translation(&snapshot.bytes, 0)[0], 2.0);
}

#[test]
fn malformed_publication_is_observable_and_can_recover() {
    let mut main = World::new();
    let a = spawn(&mut main, scene(1, 1), 1.0);
    let b = spawn(&mut main, scene(1, 1), 2.0);
    publish_transforms(&mut main);
    assert!(main
        .resource::<PublishedTransforms>()
        .last_error()
        .unwrap()
        .contains("duplicate"));
    let mut gpu = RenderTransforms::default();
    assert!(gpu
        .sync(
            main.resource::<PublishedTransforms>().snapshot.clone(),
            device(),
            &Writes::default()
        )
        .is_err());
    main.despawn(b);
    publish_transforms(&mut main);
    assert!(main
        .resource::<PublishedTransforms>()
        .last_error()
        .is_none());
    assert_eq!(
        translation(main.resource::<PublishedTransforms>().bytes(), 0)[0],
        1.0
    );
    main.remove::<Transform>(a);
    publish_transforms(&mut main);
    assert!(main
        .resource::<PublishedTransforms>()
        .last_error()
        .unwrap()
        .contains("missing global"));
}

#[test]
fn plugin_publishes_extracts_uploads_and_records_transform_bound_meshes() {
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));
    crate::tests::queue_scene_fixture(&mut app);
    let entity = spawn(app.world_mut(), scene(3, 1), 0.5);
    app.update();
    let render = app.get_sub_app(RenderApp).unwrap();
    assert_eq!(
        translation(render.world.resource::<ExtractedTransforms>().bytes(), 0)[0],
        0.5
    );
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
    let functions = render.world.resource::<DrawFunctions>().len();
    app.world_mut().increment_change_tick();
    *app.world_mut().get_mut::<Transform>(entity).unwrap() = Transform::from_xyz(0.75, 0.0, 0.0);
    app.update();
    let render = app.get_sub_app(RenderApp).unwrap();
    assert_eq!(
        render.world.resource::<DrawFunctions>().len(),
        functions,
        "draw ids must not leak on each extract"
    );
    assert_eq!(
        render
            .world
            .resource::<RenderTransforms>()
            .stats()
            .uploaded_bytes,
        48
    );
    app.update();
    assert_eq!(
        app.get_sub_app(RenderApp)
            .unwrap()
            .world
            .resource::<RenderTransforms>()
            .stats()
            .uploaded_bytes,
        0
    );
}

#[test]
fn draw_consumes_the_bound_scene_row_and_reports_unsupported_providers() {
    #[derive(Default)]
    struct Commands {
        vertices: Vec<(u32, rhi::BufferId, u64)>,
        draws: Vec<(core::ops::Range<u32>, core::ops::Range<u32>)>,
    }
    impl crate::MeshRecorder for Commands {
        fn set_pipeline(&mut self, _: rhi::RenderPipelineId) {}
        fn set_vertex_buffer(&mut self, slot: u32, buffer: rhi::BufferId, offset: u64) {
            self.vertices.push((slot, buffer, offset));
        }
        fn set_index_buffer(&mut self, _: rhi::BufferId, _: rhi::IndexFormat, _: u64) {}
        fn draw(&mut self, vertices: core::ops::Range<u32>, instances: core::ops::Range<u32>) {
            self.draws.push((vertices, instances));
        }
        fn draw_indexed(
            &mut self,
            indices: core::ops::Range<u32>,
            _: i32,
            instances: core::ops::Range<u32>,
        ) {
            self.draws.push((indices, instances));
        }
        fn draw_indexed_indirect(&mut self, _: prism_render_graph::BufferHandle, _: u64) {
            panic!("unexpected indirect draw");
        }
        fn multi_draw_indirect(
            &mut self,
            _: prism_render_graph::BufferHandle,
            _: u64,
            _: u32,
            _: u32,
        ) {
            panic!("unexpected indirect draw");
        }
    }
    struct Unsupported(rhi::RenderPipelineDescriptor);
    impl MeshPipelineProvider for Unsupported {
        fn descriptor(&self, _: &crate::MeshPipelineKey) -> rhi::RenderPipelineDescriptor {
            self.0.clone()
        }
    }
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));
    crate::tests::queue_scene_fixture(&mut app);
    spawn(app.world_mut(), scene(1, 1), 1.0);
    spawn(app.world_mut(), scene(3, 1), 0.5);
    app.update();
    let render = &mut app.sub_app_mut(RenderApp).unwrap().world;
    let work = render.resource::<crate::VisibilityWorkList>().items()[0];
    let mut request = render
        .resource::<crate::SceneRegistry>()
        .resolve_passes(
            &work,
            render.resource::<crate::RenderAssets<crate::RenderMesh>>(),
            Some(crate::RENDER_DEPTH_FORMAT),
            1,
            crate::draw::canonical::MESH,
        )
        .unwrap()
        .remove(0);
    let mut transforms = render.remove_resource::<RenderTransforms>().unwrap();
    let provider = render.resource::<MeshPipelines>();
    let unsupported = MeshPipelines::new(Unsupported(provider.descriptor(&request.variant)));
    assert!(transforms.decorate(&mut request, &unsupported).is_err());
    assert!(transforms.last_error().unwrap().contains("provider"));
    assert_eq!(transforms.stats().draws_rejected, 1);
    // Rejecting one request must not poison the prepared buffer for valid draws.
    assert!(transforms.decorate(&mut request, provider).unwrap());
    assert_eq!(request.layout.last().unwrap().array_stride, 48);
    let mut item = crate::PhaseItem {
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
    };
    let functions = render.resource::<DrawFunctions>();
    let mut commands = Commands::default();
    functions.call(item.draw, &item, &mut commands);
    assert_eq!(
        commands.vertices.last().unwrap(),
        &(1, transforms.buffer().unwrap(), 48)
    );
    assert_eq!(commands.draws, vec![(0..6, 0..1)]);
    item.scene = scene(3, 2);
    functions.call(item.draw, &item, &mut commands);
    assert_eq!(
        commands.draws.len(),
        1,
        "a reused slot cannot consume another generation's row"
    );
    render.insert_resource(transforms);
}

#[test]
fn hardware_shader_consumes_published_translation() {
    use prism_render_driver_wgpu::WgpuInstance;
    const SIZE: u32 = 16;
    let extent = rhi::Extent3d::new_2d(SIZE, SIZE);
    if prism_render_runtime::RenderRuntime::<WgpuInstance>::wgpu(extent).is_err() {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "required GPU is unavailable"
        );
        eprintln!("SKIPPED transform shader readback: no adapter");
        return;
    }
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        WgpuInstance::new(),
        RenderAppConfig::new(extent),
    ));
    crate::tests::queue_scene_fixture(&mut app);
    spawn(app.world_mut(), scene(3, 1), 0.75);
    app.update();
    let render = app.get_sub_app(RenderApp).unwrap();
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<WgpuInstance>>()
        .runtime();
    assert!(runtime.wait_idle());
    let target = render.world.resource::<crate::RenderTarget>();
    let pixels = runtime
        .queue()
        .read_texture_rgba8(target.texture(), SIZE, SIZE);
    let pixel =
        |x: u32, y: u32| &pixels[((y * SIZE + x) * 4) as usize..((y * SIZE + x + 1) * 4) as usize];
    assert_eq!(
        pixel(8, 8),
        &[0, 0, 0, 255],
        "the original center must be cleared"
    );
    assert_eq!(
        pixel(13, 8),
        &[255, 255, 255, 255],
        "the translated quad must cover its new position"
    );
}
