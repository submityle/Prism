//! Real-device tests of the production prepare/layout/binding/dispatch path.
//! Fixture rasterization only writes motion/reactive inputs; the observer only
//! reads textures. Neither substitutes a test kernel for the TAA resolve.
//! No adapter/software/unsupported feature may silently pass strict GPU mode.

use super::*;
use bevy_asset::Handle;
use bevy_camera::{CameraOutputMode, ClearColorConfig, MsaaWriteback};
use bevy_ecs::{schedule::ScheduleLabel, system::RunSystemOnce};
use bevy_math::{Mat4, UVec4};
use bevy_platform::future::block_on;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::PipelineCache,
    renderer::{CurrentView, PendingCommandBuffers, RenderAdapter, RenderDevice, RenderQueue},
    sync_world::MainEntity,
};
use bevy_shader::Shader;
use prism_math::Vec3;
use prism_render_driver_wgpu::{require_hardware_gpu, WgpuInstance};
use prism_render_shading::{resolve_taa, TaaParams};
use std::sync::{Arc, Mutex};

use crate::{
    completion::GpuCompletionTracker,
    gpu::fabric_history::{mark_scene_fabric_history_submissions, poll_scene_fabric_history},
    shading::{
        resources::prepare_visibility_buffers,
        runtime::PrismShadingSettings,
        taa::{
            bind_groups::{prepare_taa_bind_groups, ViewTaaBindGroup},
            dispatch::taa_resolve_pass,
            pipeline::create_taa_resolve_pipeline,
        },
    },
};

struct Host {
    _instance: WgpuInstance,
    adapter: RenderAdapter,
    device: RenderDevice,
    queue: RenderQueue,
}

impl Host {
    fn bridge(&self) -> SceneRhi {
        SceneRhi::from_bevy(&self.adapter, &self.device, &self.queue)
    }

    fn wait(&self) {
        self.device
            .wgpu_device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("TAA real GPU completion");
    }
}

fn host_or_skip() -> Option<Host> {
    let skip = |reason: &str| {
        assert!(!require_hardware_gpu(), "PRISM_REQUIRE_GPU=1: {reason}");
        eprintln!("SKIP_UNVERIFIED TAA production GPU test: {reason}");
        None
    };
    let instance = WgpuInstance::new();
    let Some(adapter) = instance.request_adapter(wgpu::PowerPreference::HighPerformance, false)
    else {
        return skip("no adapter");
    };
    let info = adapter.get_info();
    if !matches!(
        info.device_type,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
    ) {
        return skip(&format!("not physical hardware: {info:?}"));
    }
    let features =
        wgpu::Features::IMMEDIATES | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    if !adapter.features().contains(features) || adapter.limits().max_immediate_size < 20 {
        return skip("production TAA requires 20-byte IMMEDIATES and Rg16Float storage");
    }
    let storage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
    for (format, usages) in [
        (
            wgpu::TextureFormat::Rgba32Uint,
            storage | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        ),
        (
            wgpu::TextureFormat::Rgba16Float,
            storage | wgpu::TextureUsages::COPY_DST,
        ),
        (
            wgpu::TextureFormat::Rg16Float,
            storage | wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
        (
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
    ] {
        if !adapter
            .get_texture_format_features(format)
            .allowed_usages
            .contains(usages)
        {
            return skip(&format!("{format:?} lacks production usages {usages:?}"));
        }
    }
    eprintln!("TAA production REAL HARDWARE GPU: {info:?}");
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("TAA production test host"),
        required_features: features,
        required_limits: wgpu::Limits {
            max_immediate_size: 20,
            ..Default::default()
        },
        ..Default::default()
    }))
    .expect("physical TAA device");
    Some(Host {
        _instance: instance,
        adapter: RenderAdapter::new(adapter),
        device: RenderDevice::new(device),
        queue: RenderQueue::new(queue),
    })
}

fn render_world(host: &Host) -> World {
    // PendingCommandBuffers::finish uses Bevy's real compute task pool.
    let mut app = bevy_app::App::new();
    app.add_plugins(bevy_app::TaskPoolPlugin::default());
    let mut world = World::new();
    world.insert_resource(PrismShadingSettings {
        enable_visibility_buffer: true,
        enable_taa: true,
        ..Default::default()
    });
    world.init_resource::<TaaHistoryCache>();
    world.init_resource::<SceneFabricHistory>();
    world.init_resource::<GpuCompletionTracker>();
    world.init_resource::<PendingCommandBuffers>();
    install_host(&mut world, host);
    world
}

// Recovery replaces device-owned pipeline/binding resources, not the history
// coordinator or its monotonic completion tracker. Callers must drain the old
// queue first; this does not model device loss or abandoned queue callbacks.
fn install_host(world: &mut World, host: &Host) {
    assert!(world.resource::<PendingCommandBuffers>().is_empty());
    let history = &world.resource::<SceneFabricHistory>().lifecycle;
    assert_eq!(history.pending_submission_count(), 0);
    assert_eq!(history.pending_retirement_count(), 0);
    let tracker = world.resource::<GpuCompletionTracker>();
    assert_eq!(
        tracker.completed_value().0 + 1,
        tracker.next_submission_value().0,
        "old host queue must be drained before replacing the host"
    );
    world.insert_resource(host.device.clone());
    world.insert_resource(host.queue.clone());
    world.insert_resource(host.bridge());
    let mut cache = PipelineCache::new(host.device.clone(), true);
    let shader = Handle::<Shader>::default();
    cache.set_shader(
        shader.id(),
        Shader::from_wesl(
            include_str!("../../shaders/taa_resolve.wesl"),
            "embedded://prism_render_scene/shaders/taa_resolve.wesl",
        ),
    );
    let pipeline = create_taa_resolve_pipeline(&host.device, &cache, shader);
    cache.process_queue();
    assert!(
        cache.get_compute_pipeline(pipeline.resolve()).is_some(),
        "production TAA pipeline must compile: {:?}",
        cache.get_compute_pipeline_state(pipeline.resolve())
    );
    world.insert_resource(cache);
    world.insert_resource(pipeline);
}

fn camera(size: UVec2) -> ExtractedCamera {
    ExtractedCamera {
        target: None,
        physical_viewport_size: Some(size),
        physical_target_size: Some(size),
        viewport: None,
        schedule: bevy_core_pipeline::Core3d.intern(),
        order: 0,
        output_mode: CameraOutputMode::default(),
        msaa_writeback: MsaaWriteback::default(),
        clear_color: ClearColorConfig::default(),
        sorted_camera_index_for_target: 0,
        exposure: 1.0,
        hdr: true,
    }
}

fn spawn_view(world: &mut World, size: UVec2) -> Entity {
    let entity = world.spawn_empty().id();
    world.entity_mut(entity).insert((
        camera(size),
        ExtractedView {
            retained_view_entity: RetainedViewEntity::new(MainEntity::from(entity), None, 0),
            clip_from_view: Mat4::IDENTITY,
            world_from_view: bevy_transform::components::GlobalTransform::IDENTITY,
            clip_from_world: None,
            target_format: wgpu::TextureFormat::Rgba16Float,
            viewport: UVec4::new(0, 0, size.x, size.y),
            color_grading: Default::default(),
            invert_culling: false,
        },
    ));
    entity
}

fn prepare(world: &mut World) {
    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            prepare_visibility_buffers,
            prepare_taa_textures,
            prepare_taa_bind_groups,
        )
            .chain(),
    );
    schedule.run(world);
}

// All palette channels are exactly representable in rgba16float. Keep both
// encodings explicit so neither a half-float library nor CPU rounding can hide
// the first-frame fallback or alpha-preservation checks.
const PALETTE: [(f32, u16); 8] = [
    (0.25, 0x3400),
    (0.375, 0x3600),
    (0.5, 0x3800),
    (0.625, 0x3900),
    (0.75, 0x3a00),
    (0.875, 0x3b00),
    (1.0, 0x3c00),
    (1.125, 0x3c80),
];

fn color(x: u32, y: u32, frame: u32) -> [(f32, u16); 4] {
    [
        PALETTE[((x * 3 + y * 5 + frame * 2) % 8) as usize],
        PALETTE[((x * 5 + y * 3 + frame * 3) % 8) as usize],
        PALETTE[((x + y * 7 + frame * 5) % 8) as usize],
        PALETTE[((x + y) % 4 * 2) as usize],
    ]
}

fn motion(x: u32) -> [f32; 2] {
    match x % 4 {
        0 => [0.0, 0.0],
        1 => [0.03125, 0.015625],
        2 => [2.0, 0.0], // guaranteed offscreen, not a clipped sampler lookup
        _ => [-0.03125, -0.015625],
    }
}

fn reactive(x: u32) -> f32 {
    match x % 3 {
        0 => 0.0,
        1 => 128.0 / 255.0,
        _ => 1.0,
    }
}

fn write_inputs(host: &Host, world: &World, entity: Entity, frame: u32) -> Vec<[f32; 4]> {
    let visibility = world.get::<ViewVisibilityBuffer>(entity).unwrap();
    let size = visibility.size;
    let mut pixels = Vec::new();
    let mut bytes = Vec::new();
    for y in 0..size.y {
        for x in 0..size.x {
            let sample = color(x, y, frame);
            pixels.push(sample.map(|channel| channel.0));
            for (_, bits) in sample {
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    host.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: visibility.scene_color_texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.x * 8),
            rows_per_image: Some(size.y),
        },
        wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
    );
    // Test-only MRT writer: uses existing RENDER_ATTACHMENT usages rather than
    // adding COPY_DST to production motion/reactive allocations.
    let device = host.device.wgpu_device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("TAA fixture motion/reactive"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
    return vec4(p[i],0.0,1.0);
}
struct Output { @location(0) motion: vec2<f32>, @location(1) reactive: f32 }
@fragment fn fs(@builtin(position) p: vec4<f32>) -> Output {
    let x = u32(p.x);
    var m = vec2(0.0);
    switch x % 4u {
        case 1u: { m = vec2(0.03125,0.015625); }
        case 2u: { m = vec2(2.0,0.0); }
        case 3u: { m = vec2(-0.03125,-0.015625); }
        default: {}
    }
    var r = 0.0;
    if x % 3u == 1u { r = 0.5; }
    if x % 3u == 2u { r = 1.0; }
    return Output(m,r);
}
"#
            .into(),
        ),
    });
    let targets = [wgpu::TextureFormat::Rg16Float, wgpu::TextureFormat::R8Unorm].map(|format| {
        Some(wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("TAA input fixture only"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let attachments = [
        visibility.motion_vectors_view(),
        visibility.reactive_mask_view(),
    ]
    .map(|view| {
        Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TAA input fixture only"),
            color_attachments: &attachments,
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }
    host.queue.submit([encoder.finish()]);
    pixels
}

fn read_texture(host: &Host, view: &TextureView, size: UVec2) -> Vec<[f32; 4]> {
    // History deliberately has no COPY_SRC. Observe its actual sampled texture
    // with textureLoad, without changing production allocation usages.
    let device = host.device.wgpu_device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("TAA texture readback observer"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> out_pixels: array<vec4<f32>>;
@compute @workgroup_size(8,8) fn observe(@builtin(global_invocation_id) p: vec3<u32>) {
    let size = textureDimensions(image);
    if any(p.xy >= size) { return; }
    out_pixels[p.y*size.x+p.x] = textureLoad(image,vec2<i32>(p.xy),0);
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("TAA readback observer only"),
        layout: None,
        module: &module,
        entry_point: Some("observe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let byte_count = size.x as u64 * size.y as u64 * 16;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TAA observed rgba32"),
        size: byte_count,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TAA mapped rgba32"),
        size: byte_count,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, byte_count);
    host.queue.submit([encoder.finish()]);
    let status = Arc::new(Mutex::new(None));
    let callback_status = Arc::clone(&status);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            *callback_status.lock().unwrap() = Some(result);
        });
    host.wait();
    assert_eq!(status.lock().unwrap().take(), Some(Ok(())));
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .expect("TAA mapped readback");
    let pixels = mapped
        .chunks_exact(16)
        .map(|pixel| {
            core::array::from_fn(|channel| {
                f32::from_le_bytes(pixel[channel * 4..channel * 4 + 4].try_into().unwrap())
            })
        })
        .collect();
    drop(mapped);
    staging.unmap();
    pixels
}

fn assert_unpublished(world: &World, binding: FabricHistoryBinding) {
    assert!(world
        .resource::<SceneFabricHistory>()
        .lifecycle
        .validate_import(binding.owner, binding.binding)
        .is_err());
}

fn record(world: &mut World, entity: Entity) -> FabricHistoryBinding {
    let binding = world.get::<ViewTaa>(entity).unwrap().producer();
    let retained = world
        .get::<ExtractedView>(entity)
        .unwrap()
        .retained_view_entity;
    let cache = world.resource::<TaaHistoryCache>();
    let slot = &cache.views[&retained].slots[(binding.binding.frame as usize) & 1];
    assert_eq!(
        binding.binding.physical,
        PhysicalResource::Texture(slot.texture.id())
    );
    assert_eq!(
        world.get::<ViewTaa>(entity).unwrap().write_view().id(),
        slot.view.view().id()
    );
    assert_eq!(
        slot.view.device().wgpu_texture_view(slot.view.id()),
        **slot.view.view()
    );
    assert_unpublished(world, binding);
    let pending_before = world
        .resource::<SceneFabricHistory>()
        .lifecycle
        .pending_binding_count();
    world.insert_resource(CurrentView(entity));
    world.run_system_once(taa_resolve_pass).unwrap();
    assert!(!world.resource::<PendingCommandBuffers>().is_empty());
    assert_eq!(
        world
            .resource::<SceneFabricHistory>()
            .lifecycle
            .pending_binding_count(),
        pending_before + 1
    );
    assert_unpublished(world, binding); // recording is not completion
    binding
}

fn submit(world: &mut World, host: &Host) {
    // The publication hook must track the very queue that receives the real
    // production command buffers, not a standalone or replacement RHI queue.
    assert_eq!(
        &**world.resource::<RenderQueue>(),
        &*host.queue,
        "completion must belong to the submitting host queue"
    );
    let lifecycle = &world.resource::<SceneFabricHistory>().lifecycle;
    let has_pending =
        lifecycle.pending_binding_count() > 0 || lifecycle.pending_retirement_count() > 0;
    let expected_completion = world
        .resource::<GpuCompletionTracker>()
        .next_submission_value();
    let buffers: Vec<_> = world
        .resource_mut::<PendingCommandBuffers>()
        .finish()
        .collect();
    host.queue.submit(buffers);
    world
        .run_system_once(mark_scene_fabric_history_submissions)
        .unwrap();
    assert!(world.resource::<SceneFabricHistory>().last_error.is_none());
    assert_eq!(
        world
            .resource::<GpuCompletionTracker>()
            .next_submission_value()
            .0,
        expected_completion.0 + u64::from(has_pending),
        "production submission hook must register a real queue callback"
    );
}

fn publish(world: &mut World, host: &Host) {
    assert_eq!(
        world.resource::<RenderDevice>().wgpu_device(),
        host.device.wgpu_device(),
        "wait on the device owning the tracked submission"
    );
    host.wait();
    let tracker = world.resource::<GpuCompletionTracker>();
    assert_eq!(
        tracker.completed_value().0 + 1,
        tracker.next_submission_value().0,
        "device polling must execute all registered host-queue callbacks"
    );
    world.run_system_once(poll_scene_fabric_history).unwrap();
    let history = world.resource::<SceneFabricHistory>();
    assert!(history.last_error.is_none(), "{:?}", history.last_error);
    assert_eq!(history.lifecycle.pending_submission_count(), 0);
    assert_eq!(history.lifecycle.pending_binding_count(), 0);
    assert_eq!(history.lifecycle.pending_retirement_count(), 0);
}

fn resolve_fallback(
    world: &mut World,
    host: &Host,
    entity: Entity,
    frame: u32,
) -> FabricHistoryBinding {
    assert!(!world.get::<ViewTaa>(entity).unwrap().valid());
    let expected = write_inputs(host, world, entity, frame);
    let producer = record(world, entity);
    submit(world, host);
    let size = world.get::<ViewVisibilityBuffer>(entity).unwrap().size;
    let actual = read_texture(
        host,
        world.get::<ViewTaa>(entity).unwrap().write_view(),
        size,
    );
    assert_eq!(
        actual, expected,
        "invalid history must preserve current RGBA exactly"
    );
    assert_unpublished(world, producer); // GPU polling alone cannot publish Fabric history
    publish(world, host);
    world
        .resource::<SceneFabricHistory>()
        .lifecycle
        .validate_import(producer.owner, producer.binding)
        .unwrap();
    producer
}

fn rgb(pixel: [f32; 4]) -> Vec3 {
    Vec3::new(pixel[0], pixel[1], pixel[2])
}

fn bilinear(pixels: &[[f32; 4]], size: UVec2, uv: [f32; 2]) -> Vec3 {
    let x = uv[0] * size.x as f32 - 0.5;
    let y = uv[1] * size.y as f32 - 0.5;
    let sample = |dx: i32, dy: i32| {
        let ix = (x.floor() as i32 + dx).clamp(0, size.x as i32 - 1) as usize;
        let iy = (y.floor() as i32 + dy).clamp(0, size.y as i32 - 1) as usize;
        rgb(pixels[iy * size.x as usize + ix])
    };
    sample(0, 0).lerp(sample(1, 0), x - x.floor()).lerp(
        sample(0, 1).lerp(sample(1, 1), x - x.floor()),
        y - y.floor(),
    )
}

// Values in this fixture stay near [0.25, 1.125], where the largest binary16
// spacing is 2^-10. Allow two such steps for output quantization plus hardware
// bilinear/arithmetic differences. This is an absolute HDR-linear RGB bound,
// not a relative/image-average threshold; fallback and alpha stay exact.
const GOLDEN_RGB_TOLERANCE: f32 = 2.0 / 1024.0;

#[test]
fn production_taa_two_frame_gpu_publication_import_and_cpu_golden() {
    let Some(host) = host_or_skip() else { return };
    let mut world = render_world(&host);
    let size = UVec2::new(13, 9); // deliberately not divisible by the 8x8 dispatch tile
    let entity = spawn_view(&mut world, size);
    prepare(&mut world);
    let first_write = world.get::<ViewTaa>(entity).unwrap().write_view().id();
    let first = resolve_fallback(&mut world, &host, entity, 0);
    let previous = read_texture(
        &host,
        world.get::<ViewTaa>(entity).unwrap().write_view(),
        size,
    );
    prepare(&mut world);
    let taa = world.get::<ViewTaa>(entity).unwrap();
    assert!(taa.valid());
    assert_eq!(taa.producer().owner, first.owner);
    assert_eq!(
        taa.read_view().id(),
        first_write,
        "frame two must bind the exact published native view"
    );
    assert_ne!(taa.write_view().id(), first_write);
    let current = write_inputs(&host, &world, entity, 1);
    let visibility = world.get::<ViewVisibilityBuffer>(entity).unwrap();
    let motion_pixels = read_texture(&host, visibility.motion_vectors_view(), size);
    let reactive_pixels = read_texture(&host, visibility.reactive_mask_view(), size);
    for y in 0..size.y {
        for x in 0..size.x {
            let i = (y * size.x + x) as usize;
            assert_eq!([motion_pixels[i][0], motion_pixels[i][1]], motion(x));
            assert!((reactive_pixels[i][0] - reactive(x)).abs() < 1e-6);
        }
    }
    let second = record(&mut world, entity);
    assert_eq!(second.binding.frame, first.binding.frame + 1);
    assert_ne!(second.binding.physical, first.binding.physical);
    submit(&mut world, &host);
    assert_unpublished(&world, second);
    let actual = read_texture(
        &host,
        world.get::<ViewTaa>(entity).unwrap().write_view(),
        size,
    );
    let mut accumulated = 0;
    let mut fractional_accumulated = 0;
    let mut partial_reactive_accumulated = 0;
    let mut sign_sensitive = 0;
    let mut filter_sensitive = 0;
    let mut reactive_sensitive = 0;
    for y in 0..size.y {
        for x in 0..size.x {
            let i = (y * size.x + x) as usize;
            let m = motion(x);
            let uv = [
                (x as f32 + 0.5) / size.x as f32 - m[0],
                (y as f32 + 0.5) / size.y as f32 - m[1],
            ];
            let valid = uv.iter().all(|v| (0.0..=1.0).contains(v));
            let neighbours: [Vec3; 9] = core::array::from_fn(|tap| {
                let nx = (x as i32 + tap as i32 % 3 - 1).clamp(0, size.x as i32 - 1) as u32;
                let ny = (y as i32 + tap as i32 / 3 - 1).clamp(0, size.y as i32 - 1) as u32;
                rgb(current[(ny * size.x + nx) as usize])
            });
            let history = if valid {
                bilinear(&previous, size, uv)
            } else {
                Vec3::ZERO
            };
            let expected = resolve_taa(
                rgb(current[i]),
                history,
                &neighbours,
                &TaaParams::default(),
                valid,
                reactive(x),
            );
            // Prove this image would reject common regressions at a margin
            // well above the GPU comparison tolerance, rather than merely
            // containing nonzero motion/reactive fixture values.
            let resolve_with = |history, valid, reactive| {
                resolve_taa(
                    rgb(current[i]),
                    history,
                    &neighbours,
                    &TaaParams::default(),
                    valid,
                    reactive,
                )
            };
            let distinguishable =
                |wrong: Vec3| (expected - wrong).abs().max_element() > 8.0 * GOLDEN_RGB_TOLERANCE;
            let reversed_uv = [uv[0] + 2.0 * m[0], uv[1] + 2.0 * m[1]];
            let reversed_valid = reversed_uv.iter().all(|v| (0.0..=1.0).contains(v));
            sign_sensitive += usize::from(distinguishable(resolve_with(
                if reversed_valid {
                    bilinear(&previous, size, reversed_uv)
                } else {
                    Vec3::ZERO
                },
                reversed_valid,
                reactive(x),
            )));
            if valid {
                let nx = (uv[0] * size.x as f32).floor() as u32;
                let ny = (uv[1] * size.y as f32).floor() as u32;
                let nearest =
                    rgb(previous[(ny.min(size.y - 1) * size.x + nx.min(size.x - 1)) as usize]);
                filter_sensitive +=
                    usize::from(distinguishable(resolve_with(nearest, true, reactive(x))));
                reactive_sensitive +=
                    usize::from(distinguishable(resolve_with(history, true, 0.0)));
            }
            let error = (rgb(actual[i]) - expected).abs().max_element();
            assert!(
                actual[i].iter().all(|channel| channel.is_finite())
                    && expected.is_finite()
                    && error <= GOLDEN_RGB_TOLERANCE,
                "TAA ({x},{y}): GPU {:?}, golden {expected:?}, error {error}",
                actual[i]
            );
            assert_eq!(
                actual[i][3], current[i][3],
                "TAA must preserve current alpha"
            );
            if !valid || reactive(x) == 1.0 {
                assert_eq!(actual[i], current[i]);
            }
            if (expected - rgb(current[i])).abs().max_element() > 8.0 * GOLDEN_RGB_TOLERANCE {
                accumulated += 1;
                fractional_accumulated += usize::from(x % 4 == 1 || x % 4 == 3);
                partial_reactive_accumulated += usize::from(x % 3 == 1);
            }
        }
    }
    assert!(accumulated>10 && fractional_accumulated>0 && partial_reactive_accumulated>0,
        "fixture must distinguish real accumulation, fractional reprojection and partial reactive coverage from fallback");
    assert!(
        sign_sensitive > 0 && filter_sensitive > 0 && reactive_sensitive > 0,
        "fixture must reject reversed motion ({sign_sensitive}), nearest filtering ({filter_sensitive}), and ignored reactive masks ({reactive_sensitive})"
    );
    assert_unpublished(&world, second);
    publish(&mut world, &host);
    world
        .resource::<SceneFabricHistory>()
        .lifecycle
        .validate_import(second.owner, second.binding)
        .unwrap();
}

fn assert_retirement_pending(world: &World, owner: FabricHistoryOwner) {
    let cache = world.resource::<TaaHistoryCache>();
    assert!(cache.retired.iter().any(|entry| entry.owner == owner));
    assert_eq!(
        world
            .resource::<SceneFabricHistory>()
            .lifecycle
            .retired_before(owner, TAA_HISTORY_RESOURCE),
        None
    );
    // The retired lease still resolves to the exact real native texture in its
    // original RHI table, including when the active bridge has been replaced.
    let entry = cache
        .retired
        .iter()
        .find(|entry| entry.owner == owner)
        .unwrap();
    for slot in &entry._slots {
        assert_eq!(
            slot.texture.device().wgpu_texture(slot.texture.id()),
            **slot.texture.texture()
        );
    }
}

#[test]
fn production_taa_gpu_unpublished_and_missing_previous_frame_use_fallback() {
    let Some(host) = host_or_skip() else { return };
    let mut world = render_world(&host);
    let entity = spawn_view(&mut world, UVec2::new(13, 9));
    prepare(&mut world);
    write_inputs(&host, &world, entity, 0);
    let first_write = world.get::<ViewTaa>(entity).unwrap().write_view().id();
    let first = record(&mut world, entity);
    submit(&mut world, &host);
    host.wait();
    assert_unpublished(&world, first);
    // Even a completed GPU resolve is not importable until the production
    // Fabric poller publishes it. Do not synthesize a completion watermark.
    prepare(&mut world);
    assert_eq!(
        world.get::<ViewTaa>(entity).unwrap().read_view().id(),
        first_write
    );
    assert!(!world.get::<ViewTaa>(entity).unwrap().valid());
    publish(&mut world, &host);
    // Deliberately skip the second frame's dispatch. Published frame one must
    // not masquerade as exact previous-frame history for frame three.
    prepare(&mut world);
    assert!(!world.get::<ViewTaa>(entity).unwrap().valid());
    resolve_fallback(&mut world, &host, entity, 2);

    // A gated production dispatch must not stage another publication.
    world.resource_mut::<PrismShadingSettings>().enable_taa = false;
    world.run_system_once(taa_resolve_pass).unwrap();
    assert!(world.resource::<PendingCommandBuffers>().is_empty());
    assert_eq!(
        world
            .resource::<SceneFabricHistory>()
            .lifecycle
            .pending_binding_count(),
        0
    );
    world.run_system_once(prepare_taa_textures).unwrap();
    assert!(world.get::<ViewTaa>(entity).is_none());
    assert!(world.get::<ViewTaaBindGroup>(entity).is_none());
    submit(&mut world, &host);
    publish(&mut world, &host);
    world.run_system_once(prepare_taa_textures).unwrap();
    assert!(world.resource::<TaaHistoryCache>().retired.is_empty());
}

#[test]
fn production_taa_gpu_history_resize_removal_and_rhi_table_replacement() {
    let Some(host) = host_or_skip() else { return };
    let mut world = render_world(&host);
    let entity = spawn_view(&mut world, UVec2::new(13, 9));
    prepare(&mut world);
    let first = resolve_fallback(&mut world, &host, entity, 0);
    let size = UVec2::new(7, 11);
    world.entity_mut(entity).insert(camera(size));
    prepare(&mut world);
    let resized = world.get::<ViewTaa>(entity).unwrap().producer();
    assert_ne!(resized.owner, first.owner);
    assert!(!world.get::<ViewTaa>(entity).unwrap().valid());
    assert_retirement_pending(&world, first.owner);
    resolve_fallback(&mut world, &host, entity, 1);
    assert_eq!(
        world
            .resource::<SceneFabricHistory>()
            .lifecycle
            .retired_before(first.owner, TAA_HISTORY_RESOURCE),
        Some(resized.binding.frame)
    );
    assert!(world
        .resource::<SceneFabricHistory>()
        .lifecycle
        .validate_import(first.owner, first.binding)
        .is_err());
    prepare(&mut world); // production preparation reclaims only after publication
    assert!(world.resource::<TaaHistoryCache>().retired.is_empty());
    assert!(world.get::<ViewTaa>(entity).unwrap().valid());

    // Remove only the visibility attachment, leaving the entity and view live.
    // This caught stale ViewTaa/native bind groups surviving cache retirement.
    world.entity_mut(entity).remove::<ViewVisibilityBuffer>();
    world.run_system_once(prepare_taa_textures).unwrap();
    world.run_system_once(prepare_taa_bind_groups).unwrap();
    assert!(world.get::<ViewTaa>(entity).is_none());
    assert!(world.get::<ViewTaaBindGroup>(entity).is_none());
    assert!(world.resource::<TaaHistoryCache>().views.is_empty());
    assert_retirement_pending(&world, resized.owner);
    assert!(world.resource::<PendingCommandBuffers>().is_empty());
    // A real empty host submission is the safe retirement boundary when there
    // is no longer a view to dispatch; never inject a synthetic completion.
    submit(&mut world, &host);
    assert_retirement_pending(&world, resized.owner);
    publish(&mut world, &host);
    world.run_system_once(prepare_taa_textures).unwrap();
    assert!(world.resource::<TaaHistoryCache>().retired.is_empty());

    prepare(&mut world); // resident visibility is recreated for the live camera
    let restored = resolve_fallback(&mut world, &host, entity, 2);
    assert_ne!(restored.owner, resized.owner);
    let retained = world
        .get::<ExtractedView>(entity)
        .unwrap()
        .retained_view_entity;
    let old_table = Arc::clone(
        world.resource::<TaaHistoryCache>().views[&retained].slots[0]
            .texture
            .device(),
    );
    let replacement = host.bridge(); // same native device, distinct resource table
    assert!(!Arc::ptr_eq(&old_table, replacement.device()));
    world.insert_resource(replacement);
    prepare(&mut world);
    let replaced = world.get::<ViewTaa>(entity).unwrap().producer();
    assert_ne!(replaced.owner, restored.owner);
    assert!(!world.get::<ViewTaa>(entity).unwrap().valid());
    let cache = world.resource::<TaaHistoryCache>();
    assert!(Arc::ptr_eq(
        cache.views[&retained].slots[0].texture.device(),
        world.resource::<SceneRhi>().device()
    ));
    assert_retirement_pending(&world, restored.owner);
    resolve_fallback(&mut world, &host, entity, 3);
    assert!(world
        .resource::<SceneFabricHistory>()
        .lifecycle
        .validate_import(restored.owner, restored.binding)
        .is_err());
    prepare(&mut world);
    assert!(world.resource::<TaaHistoryCache>().retired.is_empty());
    assert!(world.get::<ViewTaa>(entity).unwrap().valid());
    assert_eq!(
        world.get::<ViewTaa>(entity).unwrap().producer().owner,
        replaced.owner
    );

    // Despawning the complete view exercises the other stale-cache path.
    world.despawn(entity);
    world.run_system_once(prepare_taa_textures).unwrap();
    assert!(world.resource::<TaaHistoryCache>().views.is_empty());
    assert_retirement_pending(&world, replaced.owner);
    submit(&mut world, &host);
    publish(&mut world, &host);
    world.run_system_once(prepare_taa_textures).unwrap();
    assert!(world.resource::<TaaHistoryCache>().retired.is_empty());
}

#[derive(Clone, Copy, Debug)]
enum PendingHistoryTransition {
    Resize,
    RemoveView,
    ReplaceTable,
}

fn pending_publication_retirement(transition: PendingHistoryTransition) {
    let Some(host) = host_or_skip() else { return };
    let mut world = render_world(&host);
    let size = UVec2::new(13, 9);
    let entity = spawn_view(&mut world, size);
    prepare(&mut world);
    let old_output = world.get::<ViewTaa>(entity).unwrap().write_view().clone();
    let old_pixels = write_inputs(&host, &world, entity, 0);
    let old = record(&mut world, entity);
    submit(&mut world, &host);
    assert_unpublished(&world, old);

    // A repeated Finish boundary with no new producer or retirement must not
    // reattach an already tracked batch or manufacture a completion token.
    // Keep the old publication pending in Fabric, regardless of whether the
    // GPU has already completed by this point; never assume hardware latency.
    let next_completion = world
        .resource::<GpuCompletionTracker>()
        .next_submission_value();
    host.queue.submit([]);
    world
        .run_system_once(mark_scene_fabric_history_submissions)
        .unwrap();
    assert!(world.resource::<SceneFabricHistory>().last_error.is_none());
    assert_eq!(
        world
            .resource::<GpuCompletionTracker>()
            .next_submission_value(),
        next_completion
    );
    assert_unpublished(&world, old);

    match transition {
        PendingHistoryTransition::Resize => {
            world.entity_mut(entity).insert(camera(UVec2::new(7, 11)));
            prepare(&mut world);
        }
        PendingHistoryTransition::RemoveView => {
            // Leave the visibility allocation and entity alive: losing only
            // ExtractedView must remove TAA state and its native bind group.
            world.entity_mut(entity).remove::<ExtractedView>();
            world.run_system_once(prepare_taa_textures).unwrap();
            world.run_system_once(prepare_taa_bind_groups).unwrap();
            assert!(world.get::<ViewVisibilityBuffer>(entity).is_some());
            assert!(world.get::<ViewTaa>(entity).is_none());
            assert!(world.get::<ViewTaaBindGroup>(entity).is_none());
            assert!(world.resource::<TaaHistoryCache>().views.is_empty());
        }
        PendingHistoryTransition::ReplaceTable => {
            // Same native queue/device, different real RHI table. Old pending
            // bindings and leases must keep their original table identity.
            world.insert_resource(host.bridge());
            prepare(&mut world);
        }
    }
    assert_retirement_pending(&world, old.owner);
    assert_unpublished(&world, old);
    assert_eq!(
        world
            .resource::<SceneFabricHistory>()
            .lifecycle
            .pending_submission_count(),
        1,
        "old producer must survive preparation until the production poller runs"
    );

    let replacement = if matches!(transition, PendingHistoryTransition::RemoveView) {
        None
    } else {
        let taa = world.get::<ViewTaa>(entity).unwrap();
        assert!(!taa.valid());
        assert_ne!(taa.producer().owner, old.owner);
        assert_ne!(taa.write_view().id(), old_output.id());
        let expected = write_inputs(&host, &world, entity, 1);
        let binding = record(&mut world, entity);
        assert_eq!(
            world
                .resource::<SceneFabricHistory>()
                .lifecycle
                .pending_binding_count(),
            2,
            "new and old real dispatches must coexist without early publication"
        );
        Some((binding, expected))
    };
    // For removal, this is a real empty host submission after the old resolve.
    // For replacement, it submits the real resolve using the new allocation.
    submit(&mut world, &host);
    assert_eq!(
        world
            .resource::<GpuCompletionTracker>()
            .next_submission_value()
            .0,
        next_completion.0 + 1,
        "new retirement must acquire a later queue completion boundary"
    );
    assert_retirement_pending(&world, old.owner);

    // These observers wait on the actual queue. Completion alone must not
    // publish the old producer or drop its retired lease before Fabric polls.
    assert_eq!(read_texture(&host, &old_output, size), old_pixels);
    if let Some((binding, expected)) = &replacement {
        let visibility = world.get::<ViewVisibilityBuffer>(entity).unwrap();
        assert_eq!(
            read_texture(
                &host,
                world.get::<ViewTaa>(entity).unwrap().write_view(),
                visibility.size,
            ),
            *expected,
            "replacement must resolve exact fallback, not read old pending history"
        );
        assert_unpublished(&world, *binding);
    }
    assert_unpublished(&world, old);
    assert_retirement_pending(&world, old.owner);
    publish(&mut world, &host);
    assert_unpublished(&world, old); // late publication must not resurrect it
    assert_eq!(
        world
            .resource::<SceneFabricHistory>()
            .lifecycle
            .retired_before(old.owner, TAA_HISTORY_RESOURCE),
        Some(old.binding.frame + 1)
    );
    if let Some((binding, _)) = replacement {
        world
            .resource::<SceneFabricHistory>()
            .lifecycle
            .validate_import(binding.owner, binding.binding)
            .unwrap();
        let new_write = world.get::<ViewTaa>(entity).unwrap().write_view().id();
        prepare(&mut world);
        let taa = world.get::<ViewTaa>(entity).unwrap();
        assert!(taa.valid());
        assert_eq!(taa.read_view().id(), new_write);
    } else {
        world.run_system_once(prepare_taa_textures).unwrap();
    }
    assert!(world.resource::<TaaHistoryCache>().retired.is_empty());
}

#[test]
fn production_taa_gpu_pending_publication_survives_resize_retirement() {
    pending_publication_retirement(PendingHistoryTransition::Resize);
}

#[test]
fn production_taa_gpu_pending_publication_survives_view_removal_retirement() {
    pending_publication_retirement(PendingHistoryTransition::RemoveView);
}

#[test]
fn production_taa_gpu_pending_publication_survives_rhi_table_retirement() {
    pending_publication_retirement(PendingHistoryTransition::ReplaceTable);
}

#[test]
fn production_taa_gpu_drained_native_device_replacement_invalidates_history() {
    let Some(old_host) = host_or_skip() else {
        return;
    };
    let Some(new_host) = host_or_skip() else {
        return;
    };
    assert_ne!(old_host.device.wgpu_device(), new_host.device.wgpu_device());
    assert_ne!(&*old_host.queue, &*new_host.queue);
    let mut world = render_world(&old_host);
    let entity = spawn_view(&mut world, UVec2::new(13, 9));
    prepare(&mut world);
    let old = resolve_fallback(&mut world, &old_host, entity, 0);
    // resolve_fallback waits and publishes on the OLD queue before any host
    // replacement. A new queue's callback cannot certify outstanding old work.
    assert_eq!(
        world
            .resource::<SceneFabricHistory>()
            .lifecycle
            .pending_submission_count(),
        0
    );
    let retained = world
        .get::<ExtractedView>(entity)
        .unwrap()
        .retained_view_entity;
    let old_table = Arc::clone(
        world.resource::<TaaHistoryCache>().views[&retained].slots[0]
            .texture
            .device(),
    );
    let old_write = world.get::<ViewTaa>(entity).unwrap().write_view().id();
    let completed_before_replacement = world.resource::<GpuCompletionTracker>().completed_value();
    install_host(&mut world, &new_host);
    assert_eq!(
        world.resource::<GpuCompletionTracker>().completed_value(),
        completed_before_replacement,
        "host replacement must not synthesize/reset completion"
    );
    prepare(&mut world);
    let taa = world.get::<ViewTaa>(entity).unwrap();
    let replacement = taa.producer();
    assert_ne!(replacement.owner, old.owner);
    assert!(!taa.valid());
    assert_ne!(taa.read_view().id(), old_write);
    assert_ne!(taa.write_view().id(), old_write);
    let cache = world.resource::<TaaHistoryCache>();
    assert!(Arc::ptr_eq(
        cache.views[&retained].slots[0].texture.device(),
        world.resource::<SceneRhi>().device()
    ));
    assert!(!Arc::ptr_eq(
        &old_table,
        world.resource::<SceneRhi>().device()
    ));
    assert!(Arc::ptr_eq(
        world
            .get::<ViewVisibilityBuffer>(entity)
            .unwrap()
            .scene_color_rhi()
            .device(),
        world.resource::<SceneRhi>().device()
    ));
    assert_retirement_pending(&world, old.owner);
    let first_new = resolve_fallback(&mut world, &new_host, entity, 1);
    assert_eq!(first_new, replacement);
    assert!(world
        .resource::<SceneFabricHistory>()
        .lifecycle
        .validate_import(old.owner, old.binding)
        .is_err());
    let new_write = world.get::<ViewTaa>(entity).unwrap().write_view().id();
    prepare(&mut world);
    assert!(world.resource::<TaaHistoryCache>().retired.is_empty());
    let taa = world.get::<ViewTaa>(entity).unwrap();
    assert!(taa.valid());
    assert_eq!(taa.read_view().id(), new_write);
    write_inputs(&new_host, &world, entity, 2);
    let second_new = record(&mut world, entity);
    submit(&mut world, &new_host);
    let pixels = read_texture(
        &new_host,
        world.get::<ViewTaa>(entity).unwrap().write_view(),
        UVec2::new(13, 9),
    );
    assert!(pixels.iter().flatten().all(|channel| channel.is_finite()));
    assert_unpublished(&world, second_new);
    publish(&mut world, &new_host);
    world
        .resource::<SceneFabricHistory>()
        .lifecycle
        .validate_import(second_new.owner, second_new.binding)
        .unwrap();
}
