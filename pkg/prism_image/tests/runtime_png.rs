//! Real producer output consumed through the existing runtime image/render seam.
//! Both backends use the existing RenderAsset uploader; hardware acceptance is
//! explicit (`--ignored`) and never silently falls back to Null/software.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use prism_asset::{AssetServer, LoadFailure, LoadScheduler, LoadState, StableGuid};
use prism_asset_build_core::{
    import::{CanonicalPngProcessor, IMAGE_ARTIFACT_KIND},
    package_runtime,
    types::{ArtifactDescriptor, ArtifactKey, AssetId, BuildRevision, Manifest, ManifestEntry},
    AssetPackageMapping, ContentAddressedStore, PackageRequest, Processor, RuntimePackage,
};
use prism_asset_container::{ContainerReader, ManifestAssetReader, OverlaySet, RuntimeManifest};
use prism_ecs::world::World;
use prism_image::{Image, ImageMipArtifact, RuntimeImageLoader};
use prism_render_app::{
    render_asset::{
        extract_assets, prepare_extracted_assets, AssetSyncReport, ExtractedAssets, RenderAssets,
    },
    GpuImage,
};
use prism_render_driver::{
    BufferId, CommandBuffer, DeviceDescriptor, Extent3d, ImageDataLayout, Instance, RenderDevice,
    RenderQueue, RequestAdapterOptions, TextureFormat, TextureWrite,
};
use prism_render_driver_null::{NullInstance, NullQueue};
use prism_tasks::{Priority, TaskPool};

const PATH: &str = "runtime://textures/odd.png";
const LOGICAL_PATH: &str = "textures/odd.png";

fn png_source(bias: u8) -> (Vec<u8>, Vec<u8>) {
    png_source_with_extent(3, 5, bias)
}

fn png_source_with_extent(width: u32, height: u32, bias: u8) -> (Vec<u8>, Vec<u8>) {
    let pixels: Vec<_> = (0..width * height)
        .flat_map(|i| [i as u8 + bias, 20, 40, 255])
        .collect();
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&pixels)
            .unwrap();
    }
    (bytes, pixels)
}

fn package(bytes: &[u8]) -> RuntimePackage {
    let directory = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(directory.path()).unwrap();
    cas.put(bytes).unwrap();
    let asset = AssetId(7);
    let manifest = Manifest::new(
        BuildRevision(1),
        [ManifestEntry {
            asset,
            descriptor: ArtifactDescriptor::for_bytes(
                ArtifactKey([7; 32]),
                IMAGE_ARTIFACT_KIND,
                2,
                16,
                vec![],
                bytes,
            ),
        }],
    )
    .unwrap();
    package_runtime(PackageRequest {
        manifest: &manifest,
        cas: &cas,
        mappings: vec![AssetPackageMapping {
            asset,
            guid: StableGuid::from_path(LOGICAL_PATH),
            container_name: "images".into(),
            priority: 7,
            chunk_priority: 0,
        }],
    })
    .unwrap()
}

fn source(package: &RuntimePackage) -> (ManifestAssetReader, RuntimeImageLoader) {
    // Consume the serialized package boundary rather than its in-memory manifest.
    let manifest = RuntimeManifest::decode(&package.runtime_manifest_bytes).unwrap();
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "images",
        7,
        ContainerReader::new(package.container_bytes("images").unwrap().to_vec()).unwrap(),
    );
    let loader = RuntimeImageLoader::new(manifest.clone(), overlays.clone());
    (ManifestAssetReader::new(manifest, overlays), loader)
}

/// Transfers the server between scheduler ownership and its normal ECS resource
/// slot; after the ticket resolves the worker may still be dropping its Arc.
fn request(
    main: &mut World,
    io_pool: &TaskPool,
    cpu_pool: &TaskPool,
    reload: bool,
) -> prism_asset::ScheduleOutcome {
    let server = Arc::new(main.remove_resource::<AssetServer>().unwrap());
    let scheduler =
        LoadScheduler::with_io_cpu_pools(Arc::clone(&server), io_pool.clone(), cpu_pool.clone(), 1);
    let ticket = if reload {
        scheduler.schedule_reload(PATH, Priority::Normal)
    } else {
        scheduler.schedule(PATH, Priority::Normal)
    }
    .unwrap();
    let result = cpu_pool.block_on(ticket);
    drop(scheduler);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut owner = server;
    loop {
        match Arc::try_unwrap(owner) {
            Ok(server) => {
                main.insert_resource(server);
                return result;
            }
            Err(shared) => {
                assert!(
                    Instant::now() < deadline,
                    "completed scheduler retained the server"
                );
                owner = shared;
                std::thread::yield_now();
            }
        }
    }
}

#[derive(Default)]
struct RecordingQueue {
    inner: NullQueue,
    writes: Mutex<Vec<(TextureWrite, Vec<u8>, ImageDataLayout)>>,
}

impl RenderQueue for RecordingQueue {
    fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]) {
        self.inner.write_buffer(buffer, offset, data);
    }

    fn write_texture(&self, destination: TextureWrite, data: &[u8], layout: ImageDataLayout) {
        self.inner.write_texture(destination, data, layout);
        self.writes
            .lock()
            .unwrap()
            .push((destination, data.to_vec(), layout));
    }

    fn submit(&self, buffers: &[CommandBuffer]) {
        self.inner.submit(buffers);
    }
}

fn frame(
    main: &mut World,
    render: &mut World,
    device: &dyn RenderDevice,
    queue: &dyn RenderQueue,
) -> AssetSyncReport {
    extract_assets::<GpuImage>(main, render);
    let mut extracted = render
        .remove_resource::<ExtractedAssets<GpuImage>>()
        .unwrap_or_default();
    let mut prepared = render
        .remove_resource::<RenderAssets<GpuImage>>()
        .unwrap_or_default();
    let report = prepare_extracted_assets(&mut extracted, &mut prepared, device, queue);
    render.insert_resource(extracted);
    render.insert_resource(prepared);
    report
}

#[test]
fn png_package_async_image_render_lifecycle() {
    let (png, pixels) = png_source(0);
    let artifact = CanonicalPngProcessor.process(&[&png]).unwrap();
    let expected = ImageMipArtifact::decode(&artifact)
        .unwrap()
        .to_image()
        .unwrap();
    assert_eq!(expected.data(), pixels);
    assert_eq!(expected.mip_level_count(), 4);
    let packaged = package(&artifact);
    let (mounted, loader) = source(&packaged);
    let server = AssetServer::new();
    server.register_loader(loader);
    server.mount("runtime", Arc::new(mounted), 0);
    let mut main = World::new();
    let mut render = World::new();
    main.insert_resource(server);
    let io_pool = TaskPool::with_threads(1);
    let cpu_pool = TaskPool::with_threads(1);
    let handle = request(&mut main, &io_pool, &cpu_pool, false).unwrap();
    let id = handle.id().typed::<Image>().unwrap();
    assert_eq!(
        main.resource::<AssetServer>().load_state(handle.id()),
        LoadState::Loaded
    );
    assert_eq!(
        main.resource::<AssetServer>()
            .with_asset(id, |image| image.data().to_vec()),
        Some(pixels)
    );

    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .unwrap();
    let (device, _) = instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .unwrap();
    let queue = RecordingQueue::default();
    assert_eq!(frame(&mut main, &mut render, &device, &queue).prepared, 1);
    let first = render.resource::<RenderAssets<GpuImage>>().get(id).unwrap();
    assert_eq!(first.extent, Extent3d::new_2d(3, 5));
    assert_eq!(first.format, TextureFormat::Rgba8Unorm);
    assert_eq!(first.mip_level_count, 4);
    let first_texture = first.texture;
    let first_sampler = first.sampler;
    let writes = queue.writes.lock().unwrap();
    assert_eq!(writes.len(), 4);
    for (level, (destination, data, layout)) in writes.iter().enumerate() {
        let level = level as u32;
        let extent = expected.mip_extent(level).unwrap();
        assert_eq!(destination.texture, first_texture);
        assert_eq!(destination.mip_level, level);
        assert_eq!(destination.origin, [0, 0, 0]);
        assert_eq!(destination.size, extent);
        assert_eq!(data, expected.mip_data(level).unwrap());
        assert_eq!(layout.offset, 0);
        assert_eq!(layout.bytes_per_row, Some(extent.width * 4));
        assert_eq!(layout.rows_per_image, Some(extent.height));
    }
    drop(writes);
    assert_eq!(device.live_resource_count(), 3);
    assert!(!frame(&mut main, &mut render, &device, &queue).changed());
    assert_eq!(queue.writes.lock().unwrap().len(), 4, "no duplicate upload");

    // A valid container carrying an uncooked PNG must fail the image decoder,
    // not upload encoded PNG bytes or discard the last-known-good image.
    main.resource::<AssetServer>()
        .mount("runtime", Arc::new(source(&package(&png)).0), 0);
    assert!(matches!(
        request(&mut main, &io_pool, &cpu_pool, true),
        Err(LoadFailure::AssemblyFailed(message))
            if message.contains("invalid canonical image artifact header")
    ));
    assert_eq!(
        main.resource::<AssetServer>()
            .with_asset(id, |image| image.data().to_vec()),
        Some(expected.data().to_vec())
    );
    assert!(!frame(&mut main, &mut render, &device, &queue).changed());
    assert_eq!(
        render
            .resource::<RenderAssets<GpuImage>>()
            .get(id)
            .unwrap()
            .texture,
        first_texture
    );
    assert_eq!(queue.writes.lock().unwrap().len(), 4);
    assert_eq!(device.live_resource_count(), 3);

    let (changed_png, changed_pixels) = png_source(80);
    let changed_artifact = CanonicalPngProcessor.process(&[&changed_png]).unwrap();
    let changed_image = ImageMipArtifact::decode(&changed_artifact)
        .unwrap()
        .to_image()
        .unwrap();
    main.resource::<AssetServer>().mount(
        "runtime",
        Arc::new(source(&package(&changed_artifact)).0),
        0,
    );
    let reloaded = request(&mut main, &io_pool, &cpu_pool, true).unwrap();
    assert_eq!(reloaded.id(), handle.id());
    assert_eq!(
        main.resource::<AssetServer>()
            .with_asset(id, |image| image.data().to_vec()),
        Some(changed_pixels)
    );
    assert_eq!(frame(&mut main, &mut render, &device, &queue).prepared, 1);
    let second = render.resource::<RenderAssets<GpuImage>>().get(id).unwrap();
    assert_ne!(second.texture, first_texture);
    assert_eq!(second.sampler, first_sampler);
    let writes = queue.writes.lock().unwrap();
    assert_eq!(writes.len(), 8);
    for (level, (destination, data, layout)) in writes[4..].iter().enumerate() {
        let level = level as u32;
        let extent = changed_image.mip_extent(level).unwrap();
        assert_eq!(destination.texture, second.texture);
        assert_eq!(destination.mip_level, level);
        assert_eq!(destination.origin, [0, 0, 0]);
        assert_eq!(destination.size, extent);
        assert_eq!(data, changed_image.mip_data(level).unwrap());
        assert_eq!(layout.offset, 0);
        assert_eq!(layout.bytes_per_row, Some(extent.width * 4));
        assert_eq!(layout.rows_per_image, Some(extent.height));
    }
    drop(writes);
    assert_eq!(device.live_resource_count(), 3);

    drop(reloaded);
    drop(handle);
    main.resource::<AssetServer>().collect_releases();
    assert_eq!(frame(&mut main, &mut render, &device, &queue).released, 1);
    assert!(!render.resource::<RenderAssets<GpuImage>>().contains(id));
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn raw_png_is_not_a_runtime_image_artifact() {
    let (png, _) = png_source(0);
    let (mounted, loader) = source(&package(&png));
    let server = AssetServer::new();
    server.register_loader(loader);
    server.mount("runtime", Arc::new(mounted), 0);
    let handle = server.load::<Image>(PATH);
    assert!(server.load_state(handle.id().untyped()).is_failed());
    assert!(server.with_asset(handle.id(), Clone::clone).is_none());
}

/// Exercise shader sampling, not a copy of the uploaded texture (GpuImage does
/// not expose COPY_SRC). Every mip is drawn into its own readback attachment.
fn assert_sampled_mips(
    device: &prism_render_driver_wgpu::WgpuDevice,
    queue: &prism_render_driver_wgpu::WgpuQueue,
    prepared: &GpuImage,
    expected: &Image,
) {
    use prism_render_driver::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
        BindingResource, BindingType, Color, ColorAttachment, ColorTargetState, CommandEncoder,
        FragmentState, LoadOp, MultisampleState, PipelineLayoutDescriptor, PrimitiveState,
        RenderCommand, RenderPassDescriptor, RenderPipelineDescriptor, SamplerBindingType,
        ShaderModuleDescriptor, ShaderStages, StoreOp, TextureDescriptor, TextureSampleType,
        TextureUsages, TextureViewDimension, VertexState,
    };

    assert_eq!(prepared.extent, expected.extent());
    assert_eq!(prepared.mip_level_count, expected.mip_level_count());
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("runtime-png-sample-layout".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("runtime-png-sample".into()),
        layout: Some(layout),
        entries: vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(prepared.view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(prepared.sampler),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("runtime-png-sample".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: vec![],
    });
    for level in 0..expected.mip_level_count() {
        let extent = expected.mip_extent(level).unwrap();
        let shader = format!(
            r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {{
    var corners = array<vec2<f32>, 6>(
        vec2(-1., -1.), vec2(1., -1.), vec2(1., 1.),
        vec2(-1., -1.), vec2(1., 1.), vec2(-1., 1.));
    return vec4(corners[index], 0., 1.);
}}
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{
    let uv = p.xy / vec2<f32>({width}.0, {height}.0);
    return textureSampleLevel(tex, samp, uv, {level}.0);
}}
"#,
            width = extent.width,
            height = extent.height,
        );
        let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(Box::leak(
            shader.into_boxed_str(),
        )));
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("runtime-png-sample".into()),
            layout: Some(pipeline_layout),
            vertex: VertexState {
                module,
                entry_point: "vs".into(),
                buffers: vec![],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module,
                entry_point: "fs".into(),
                targets: vec![Some(ColorTargetState::opaque(TextureFormat::Rgba8Unorm))],
            }),
        });
        let target = device.create_texture(&TextureDescriptor::new_2d(
            extent,
            TextureFormat::Rgba8Unorm,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let view = device.create_texture_view(target, &Default::default());
        let mut encoder = CommandEncoder::labeled("runtime-png-sample");
        encoder.push_render_pass(
            RenderPassDescriptor {
                label: Some("runtime-png-sample".into()),
                color_attachments: vec![Some(ColorAttachment {
                    view,
                    resolve_target: None,
                    load: LoadOp::Clear(Color::TRANSPARENT),
                    store: StoreOp::Store,
                })],
                depth_stencil_attachment: None,
            },
            vec![
                RenderCommand::SetPipeline(pipeline),
                RenderCommand::SetBindGroup {
                    index: 0,
                    bind_group: group,
                    dynamic_offsets: vec![],
                },
                RenderCommand::Draw {
                    vertices: 0..6,
                    instances: 0..1,
                },
            ],
        );
        queue.submit(&[encoder.finish()]);
        assert!(queue.wait_idle(), "mip {level} sampling must finish");
        let sampled = queue
            .read_texture_rgba8(target, extent.width, extent.height)
            .expect("PNG mip GPU readback must complete");
        device.destroy_texture_view(view);
        device.destroy_texture(target);
        device.destroy_render_pipeline(pipeline);
        device.destroy_shader_module(module);
        assert_eq!(sampled, expected.mip_data(level).unwrap(), "mip {level}");
    }
    device.destroy_bind_group(group);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_bind_group_layout(layout);
}

#[test]
#[ignore = "explicit physical GPU acceptance; fails if no integrated/discrete adapter exists"]
fn png_package_async_image_real_gpu_lifecycle() {
    let instance = prism_render_driver_wgpu::WgpuInstance::new();
    let (device, queue) = instance
        .request_default_hardware_device()
        .expect("PNG acceptance requires a physical GPU, not software/Null");
    eprintln!(
        "PNG physical GPU: {:?}",
        device.wgpu_device().adapter_info()
    );

    // Power-of-two dimensions match native GPU mip geometry. The odd-sized
    // producer's ceil-halved chain is covered separately by the Null test.
    let (png, pixels) = png_source_with_extent(4, 4, 0);
    let artifact = CanonicalPngProcessor.process(&[&png]).unwrap();
    let expected = ImageMipArtifact::decode(&artifact)
        .unwrap()
        .to_image()
        .unwrap();
    assert_eq!(expected.data(), pixels);
    assert_eq!(expected.mip_level_count(), 3);
    let (mounted, loader) = source(&package(&artifact));
    let server = AssetServer::new();
    server.register_loader(loader);
    server.mount("runtime", Arc::new(mounted), 0);
    let mut main = World::new();
    let mut render = World::new();
    main.insert_resource(server);
    let io_pool = TaskPool::with_threads(1);
    let cpu_pool = TaskPool::with_threads(1);
    let handle = request(&mut main, &io_pool, &cpu_pool, false).unwrap();
    let id = handle.id().typed::<Image>().unwrap();
    assert_eq!(frame(&mut main, &mut render, &device, &queue).prepared, 1);
    let first = render.resource::<RenderAssets<GpuImage>>().get(id).unwrap();
    let first_texture = first.texture;
    let first_sampler = first.sampler;
    assert_sampled_mips(&device, &queue, first, &expected);
    assert!(!frame(&mut main, &mut render, &device, &queue).changed());

    main.resource::<AssetServer>()
        .mount("runtime", Arc::new(source(&package(&png)).0), 0);
    assert!(matches!(
        request(&mut main, &io_pool, &cpu_pool, true),
        Err(LoadFailure::AssemblyFailed(message))
            if message.contains("invalid canonical image artifact header")
    ));
    assert!(!frame(&mut main, &mut render, &device, &queue).changed());
    let retained = render.resource::<RenderAssets<GpuImage>>().get(id).unwrap();
    assert_eq!(retained.texture, first_texture);
    assert_sampled_mips(&device, &queue, retained, &expected);

    let (changed_png, changed_pixels) = png_source_with_extent(4, 4, 80);
    let changed_artifact = CanonicalPngProcessor.process(&[&changed_png]).unwrap();
    let changed = ImageMipArtifact::decode(&changed_artifact)
        .unwrap()
        .to_image()
        .unwrap();
    main.resource::<AssetServer>().mount(
        "runtime",
        Arc::new(source(&package(&changed_artifact)).0),
        0,
    );
    let reloaded = request(&mut main, &io_pool, &cpu_pool, true).unwrap();
    assert_eq!(reloaded.id(), handle.id());
    assert_eq!(
        main.resource::<AssetServer>()
            .with_asset(id, |image| image.data().to_vec()),
        Some(changed_pixels)
    );
    assert_eq!(frame(&mut main, &mut render, &device, &queue).prepared, 1);
    let second = render.resource::<RenderAssets<GpuImage>>().get(id).unwrap();
    assert_ne!(second.texture, first_texture);
    assert_eq!(second.sampler, first_sampler);
    assert_sampled_mips(&device, &queue, second, &changed);

    drop(reloaded);
    drop(handle);
    main.resource::<AssetServer>().collect_releases();
    assert_eq!(frame(&mut main, &mut render, &device, &queue).released, 1);
    assert!(!render.resource::<RenderAssets<GpuImage>>().contains(id));
    assert!(queue.wait_idle(), "release frame must finish");
}
