//! Runs the real prepare and resident systems against a real GPU render world.

use super::*;
use bevy_camera::{CameraOutputMode, ClearColorConfig, MsaaWriteback};
use bevy_ecs::{schedule::ScheduleLabel, system::RunSystemOnce};
use bevy_platform::future::block_on;
use bevy_render::{
    camera::ExtractedCamera,
    renderer::{CurrentView, PendingCommandBuffers},
    texture::TextureCache,
};
use bevy_tasks::{ComputeTaskPool, TaskPool};

fn device() -> Option<(wgpu::Device, wgpu::Queue, RenderDevice)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: Default::default(),
    });
    let adapter = match block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "required GPU: {error}"
            );
            eprintln!("resident ECS GPU skipped: {error}");
            return None;
        }
    };
    eprintln!("resident ECS adapter: {:?}", adapter.get_info());
    let limits = adapter.limits();
    let hdr_usage = adapter
        .get_texture_format_features(SCENE_COLOR_FORMAT)
        .allowed_usages;
    let visibility_usage = adapter
        .get_texture_format_features(VISIBILITY_ID_FORMAT)
        .allowed_usages;
    let motion_usage = adapter
        .get_texture_format_features(super::super::resources::MOTION_VECTOR_FORMAT)
        .allowed_usages;
    if let Some(error) = capability_error(
        adapter.features(),
        limits,
        hdr_usage,
        visibility_usage,
        motion_usage,
    ) {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "PRISM_REQUIRE_GPU set: resident ECS capability gate failed: {error}"
        );
        eprintln!("resident ECS GPU skipped: {error}");
        return None;
    }
    // prepare_visibility_buffers allocates the production Rg16Float storage
    // texture as well as the composition targets. Match Bevy's format feature
    // request; default WebGPU features alone do not admit that storage format.
    let descriptor = wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        ..Default::default()
    };
    let (device, queue) = match block_on(adapter.request_device(&descriptor)) {
        Ok(device_queue) => device_queue,
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "PRISM_REQUIRE_GPU set: resident ECS device request failed: {error}"
            );
            eprintln!("resident ECS GPU skipped: device request failed: {error}");
            return None;
        }
    };
    Some((device.clone(), queue, RenderDevice::from(device)))
}

fn capability_error(
    features: wgpu::Features,
    limits: Limits,
    hdr_usage: TextureUsages,
    visibility_usage: TextureUsages,
    motion_usage: TextureUsages,
) -> Option<String> {
    let required_visibility = TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    let required_storage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    if !features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
        return Some("Rg16Float storage needs TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES".into());
    }
    if limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_size_y < 8
    {
        return Some(format!(
            "resident ECS composition needs compute workgroup 64x1 and HDR bridge 8x8; adapter limits are {limits:?}"
        ));
    }
    if !hdr_usage.contains(required_storage) {
        return Some(format!(
            "Rgba16Float lacks TEXTURE_BINDING|STORAGE_BINDING; allowed usages are {hdr_usage:?}"
        ));
    }
    if !visibility_usage.contains(required_visibility) {
        return Some(format!(
            "Rgba32Uint lacks TEXTURE_BINDING|RENDER_ATTACHMENT; allowed usages are {visibility_usage:?}"
        ));
    }
    if !motion_usage.contains(required_storage | TextureUsages::RENDER_ATTACHMENT) {
        return Some(format!(
            "Rg16Float lacks TEXTURE_BINDING|STORAGE_BINDING|RENDER_ATTACHMENT; allowed usages are {motion_usage:?}"
        ));
    }
    None
}

#[test]
fn resident_ecs_capability_gate_rejects_undersized_workgroups() {
    let mut limits = Limits::default();
    limits.max_compute_workgroup_size_x = 63;
    let error = capability_error(
        wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        limits,
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::RENDER_ATTACHMENT,
    )
    .expect("workgroup x=63 must be rejected before device use");
    assert!(error.contains("workgroup"));
}

#[test]
fn resident_ecs_capability_gate_requires_motion_storage_feature_and_usage() {
    let storage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    let visibility = TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    let feature = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    let missing_feature = capability_error(
        wgpu::Features::empty(),
        Limits::default(),
        storage,
        visibility,
        storage | TextureUsages::RENDER_ATTACHMENT,
    )
    .unwrap();
    assert!(missing_feature.contains("TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES"));
    let missing_usage = capability_error(
        feature,
        Limits::default(),
        storage,
        visibility,
        TextureUsages::TEXTURE_BINDING,
    )
    .unwrap();
    assert!(missing_usage.contains("Rg16Float lacks"));
    assert!(capability_error(feature, Limits::default(), storage, visibility, storage).is_some());
    assert!(capability_error(
        feature,
        Limits::default(),
        storage,
        visibility,
        storage | TextureUsages::RENDER_ATTACHMENT,
    )
    .is_none());
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

/// Real view preparation must cache targets, render HDR once, reject stale
/// frames, and invalidate the old bind group when a view/response is resized.
#[test]
fn resident_composition_ecs_prepares_executes_and_invalidates_targets() {
    let Some((device, queue, render_device)) = device() else {
        return;
    };
    ComputeTaskPool::get_or_init(TaskPool::new);
    let size = UVec2::new(3, 2);
    let pipelines = ResidentCompositionPipelines::new(&render_device);
    let responses = ViewResponseComposition::new(&render_device, &pipelines, size, 0).unwrap();
    let mut world = World::new();
    world.insert_resource(render_device.clone());
    world.insert_resource(pipelines);
    world.insert_resource(PrismShadingSettings {
        enable_visibility_buffer: true,
        frame_index: 9,
        ..Default::default()
    });
    world.init_resource::<TextureCache>();
    world.init_resource::<PendingCommandBuffers>();
    let view = world.spawn((camera(size), responses)).id();
    world.insert_resource(CurrentView(view));
    world
        .run_system_once(super::super::resources::prepare_visibility_buffers)
        .unwrap();
    world
        .run_system_once(prepare_response_composition_targets)
        .unwrap();
    let target_id = world
        .get::<ResponseCompositionTarget>(view)
        .unwrap()
        .group
        .id();
    world
        .run_system_once(prepare_response_composition_targets)
        .unwrap();
    assert_eq!(
        world
            .get::<ResponseCompositionTarget>(view)
            .unwrap()
            .group
            .id(),
        target_id
    );

    let responses = world.get::<ViewResponseComposition>(view).unwrap();
    let count = responses.layout.pixel_count() as usize;
    queue.write_buffer(
        &responses.left,
        0,
        bytemuck::cast_slice(&vec![GpuResponseOutput { words: [1.0; 16] }; count]),
    );
    queue.write_buffer(
        &responses.right,
        0,
        bytemuck::cast_slice(&vec![GpuResponseOutput { words: [5.0; 16] }; count]),
    );
    queue.write_buffer(
        &responses.controls,
        0,
        bytemuck::cast_slice(&vec![
            GpuCompositionControl {
                mask: 0.25,
                channels: 31,
                shared_alpha: 0.75,
                alpha_rule: 0,
            };
            count
        ]),
    );
    // Production visibility textures are render attachments, not COPY_DST.
    // Clear them through that actual usage to mark every fixture pixel covered.
    let visibility = world.get::<ViewVisibilityBuffer>(view).unwrap();
    let (ids, metadata, _) = visibility.feedback_sources();
    let mut clear = device.create_command_encoder(&Default::default());
    for texture in [ids, metadata] {
        let texture_view = texture.create_view(&Default::default());
        let attachment = wgpu::RenderPassColorAttachment {
            view: &texture_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        };
        let _pass = clear.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(attachment)],
            ..Default::default()
        });
    }
    queue.submit([clear.finish()]);
    world
        .get_mut::<ViewResponseComposition>(view)
        .unwrap()
        .publish(9);
    world.run_system_once(response_composition_pass).unwrap();
    assert_eq!(world.resource::<PendingCommandBuffers>().len(), 1);
    queue.submit(world.resource_mut::<PendingCommandBuffers>().finish());
    world.run_system_once(response_composition_pass).unwrap();
    assert!(world.resource::<PendingCommandBuffers>().is_empty());
    assert!(world
        .get::<ViewResponseComposition>(view)
        .unwrap()
        .last_error()
        .is_none());

    // Read through TEXTURE_BINDING, the same consumer usage as the production
    // composite; scene_color deliberately does not have COPY_SRC usage.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resident ECS HDR readback"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> pixels: array<vec4<f32>>;
@compute @workgroup_size(1)
fn read_hdr(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(hdr);
    if id.x < size.x && id.y < size.y {
        pixels[id.y * size.x + id.x] = textureLoad(hdr, vec2<i32>(id.xy), 0);
    }
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("resident ECS HDR readback"),
        layout: None,
        module: &shader,
        entry_point: Some("read_hdr"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bytes = count as u64 * 16;
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let hdr_view = world
        .get::<ViewVisibilityBuffer>(view)
        .unwrap()
        .scene_color_texture()
        .create_view(&Default::default());
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hdr_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(size.x, size.y, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, bytes);
    queue.submit([encoder.finish()]);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.expect("map HDR"));
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = readback.slice(..).get_mapped_range().unwrap();
    assert_eq!(
        bytemuck::cast_slice::<u8, [f32; 4]>(&mapped),
        vec![[10.0, 10.0, 10.0, 0.75]; count]
    );
    drop(mapped);
    readback.unmap();

    let resized = UVec2::new(4, 2);
    world.entity_mut(view).insert(camera(resized));
    world
        .run_system_once(super::super::resources::prepare_visibility_buffers)
        .unwrap();
    world
        .run_system_once(prepare_response_composition_targets)
        .unwrap();
    assert!(world.get::<ResponseCompositionTarget>(view).is_none());
    assert!(matches!(
        world
            .get::<ViewResponseComposition>(view)
            .unwrap()
            .last_error(),
        Some(CompositionResourceError::InvalidTarget { .. })
    ));
    let replacement = ViewResponseComposition::new(
        &render_device,
        world.resource::<ResidentCompositionPipelines>(),
        resized,
        0,
    )
    .unwrap();
    world.entity_mut(view).insert(replacement);
    world
        .run_system_once(prepare_response_composition_targets)
        .unwrap();
    assert_ne!(
        world
            .get::<ResponseCompositionTarget>(view)
            .unwrap()
            .group
            .id(),
        target_id
    );
    assert!(world
        .get::<ViewResponseComposition>(view)
        .unwrap()
        .last_error()
        .is_none());
    world
        .get_mut::<ViewResponseComposition>(view)
        .unwrap()
        .publish(8);
    world.run_system_once(response_composition_pass).unwrap();
    assert!(world.resource::<PendingCommandBuffers>().is_empty());
}
