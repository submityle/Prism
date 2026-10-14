//! Real GPU producer -> resident compositor -> shared HDR readback.
//! The parent compositor suite owns channel/rule parity; this suite verifies
//! allocation, command ordering, independent auxiliary binding and HDR export.

use bevy_math::UVec2;
use bevy_platform::future::block_on;
use bevy_render::{render_resource::Texture, renderer::RenderDevice};
use prism_render_scene::{
    composition::{
        CompositionPlane, CompositionResourceError, ResidentCompositionPipelines,
        ResponseCompositionTarget, ViewResponseComposition,
    },
    response_gpu::{GpuAuxiliaryMerge, GpuResponseOutput},
};

// This is a test producer, not another compositor: arms are written independently
// from a shared fixture input. Production composition/export are used unchanged.
const PRODUCER: &str = r#"
struct Response { words: array<f32, 16>, }
struct Control { mask: f32, channels: u32, shared_alpha: f32, alpha_rule: u32, }
struct Auxiliary {
    left: vec4<u32>, right: vec4<u32>, shared_value: vec4<u32>,
    kind: u32, rule: u32, mask: f32, reserved: u32,
}
@group(0) @binding(0) var<storage, read_write> left: array<Response>;
@group(0) @binding(1) var<storage, read_write> right: array<Response>;
@group(0) @binding(2) var<storage, read_write> controls: array<Control>;
@group(0) @binding(3) var<storage, read_write> auxiliary: array<Auxiliary>;
@compute @workgroup_size(64)
fn produce(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i < arrayLength(&left) {
        for (var c = 0u; c < 5u; c += 1u) {
            let offset = 3u * c;
            let variation = f32(i % 3u) * 4.0;
            left[i].words[offset] = 1.0 + variation;
            right[i].words[offset] = 5.0 + variation;
            left[i].words[offset + 1u] = 2.0;
            right[i].words[offset + 1u] = 6.0;
            left[i].words[offset + 2u] = 3.0;
            right[i].words[offset + 2u] = 7.0;
        }
        left[i].words[15] = 0.25;
        right[i].words[15] = 1.0;
        controls[i] = Control(0.25, 31u, 0.75, 0u);
    }
    if i < arrayLength(&auxiliary) {
        auxiliary[i] = Auxiliary(vec4<u32>(i, 0u, 0u, 0u),
            vec4<u32>(0x80000000u, 0u, 0u, 0u), vec4<u32>(0u),
            3u, 7u, 0.25, 0u);
    }
}
"#;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue, RenderDevice)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: Default::default(),
    });
    let adapter = match block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "resident composition requires a GPU: {error}"
            );
            eprintln!("resident composition skipped: {error}");
            return None;
        }
    };
    eprintln!("resident composition adapter: {:?}", adapter.get_info());
    let limits = adapter.limits();
    let hdr_usage = adapter
        .get_texture_format_features(wgpu::TextureFormat::Rgba16Float)
        .allowed_usages;
    let visibility_usage = adapter
        .get_texture_format_features(wgpu::TextureFormat::Rgba32Uint)
        .allowed_usages;
    if let Some(error) = capability_error(limits, hdr_usage, visibility_usage) {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "PRISM_REQUIRE_GPU set: resident composition capability gate failed: {error}"
        );
        eprintln!("resident composition skipped: {error}");
        return None;
    }
    let (device, queue) = match block_on(adapter.request_device(&Default::default())) {
        Ok(device_queue) => device_queue,
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "PRISM_REQUIRE_GPU set: resident composition device request failed: {error}"
            );
            eprintln!("resident composition skipped: device request failed: {error}");
            return None;
        }
    };
    let render_device = RenderDevice::from(device.clone());
    Some((device, queue, render_device))
}

fn capability_error(
    limits: wgpu::Limits,
    hdr_usage: wgpu::TextureUsages,
    visibility_usage: wgpu::TextureUsages,
) -> Option<String> {
    let required_visibility =
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT;
    if limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_size_y < 8
    {
        return Some(format!(
            "resident composition needs compute workgroup 64x1 and HDR bridge 8x8; adapter limits are {limits:?}"
        ));
    }
    if !hdr_usage.contains(wgpu::TextureUsages::STORAGE_BINDING) {
        return Some(format!(
            "Rgba16Float lacks STORAGE_BINDING; allowed usages are {hdr_usage:?}"
        ));
    }
    if !visibility_usage.contains(required_visibility) {
        return Some(format!(
            "Rgba32Uint lacks TEXTURE_BINDING|RENDER_ATTACHMENT; allowed usages are {visibility_usage:?}"
        ));
    }
    None
}

#[test]
fn resident_composition_capability_gate_rejects_missing_hdr_storage_support() {
    let error = capability_error(
        wgpu::Limits::default(),
        wgpu::TextureUsages::TEXTURE_BINDING,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
    )
    .expect("missing HDR storage support must be rejected before allocation");
    assert!(error.contains("Rgba16Float"));
}

fn texture(device: &RenderDevice, size: UVec2, format: wgpu::TextureFormat) -> Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resident composition test target"),
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn upload_texture(queue: &wgpu::Queue, target: &Texture, bytes: &[u8], stride: u32) {
    queue.write_texture(
        target.as_image_copy(),
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(target.width() * stride),
            rows_per_image: Some(target.height()),
        },
        target.size(),
    );
}

fn readback(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resident composition readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read<T: bytemuck::Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<T> {
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.expect("map readback"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("complete resident commands");
    let mapped = buffer.slice(..).get_mapped_range().expect("mapped data");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}

// All fixture values are normal, exactly representable positive binary16.
fn normal_half(value: u16) -> f32 {
    let exponent = u32::from((value >> 10) & 31);
    assert!((1..31).contains(&exponent));
    f32::from_bits(((exponent + 112) << 23) | (u32::from(value & 1023) << 13))
}

/// Odd dimensions cross both 64-wide and 8x8 boundaries. Two frames verify that
/// in-place composition is never repeated without fresh, fully rewritten arms.
#[test]
fn gpu_producers_flow_through_resident_composition_into_hdr() {
    let Some((device, queue, render_device)) = gpu() else {
        return;
    };
    let size = UVec2::new(17, 9);
    let pipelines = ResidentCompositionPipelines::new(&render_device);
    let mut responses =
        ViewResponseComposition::new(&render_device, &pipelines, size, 65).expect("admission");
    let hdr = texture(&render_device, size, wgpu::TextureFormat::Rgba16Float);
    let ids = texture(&render_device, size, wgpu::TextureFormat::Rgba32Uint);
    let metadata = texture(&render_device, size, wgpu::TextureFormat::Rgba32Uint);
    let target = ResponseCompositionTarget::new(
        &render_device,
        &pipelines,
        &responses,
        &hdr,
        &ids,
        &metadata,
    )
    .expect("matching targets");
    let pixels = responses.layout().pixel_count() as usize;
    let ids_data: Vec<[u32; 4]> = (0..pixels)
        .map(|i| [if i % 11 == 0 { u32::MAX } else { i as u32 }, 0, 0, 0])
        .collect();
    let metadata_data: Vec<[u32; 4]> = (0..pixels)
        .map(|i| [if i % 13 == 0 { u32::MAX } else { 0 }, 0, 0, 0])
        .collect();
    upload_texture(&queue, &ids, bytemuck::cast_slice(&ids_data), 16);
    upload_texture(&queue, &metadata, bytemuck::cast_slice(&metadata_data), 16);
    let initial_hdr = vec![[0x3c00u16, 0x4000, 0x4200, 0x3c00]; pixels];
    upload_texture(&queue, &hdr, bytemuck::cast_slice(&initial_hdr), 8);

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("GPU test split producer"),
        source: wgpu::ShaderSource::Wgsl(PRODUCER.into()),
    });
    let producer = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("GPU test split producer"),
        layout: None,
        module: &module,
        entry_point: Some("produce"),
        compilation_options: Default::default(),
        cache: None,
    });
    let entries: Vec<_> = [
        CompositionPlane::Left,
        CompositionPlane::Right,
        CompositionPlane::Controls,
        CompositionPlane::Auxiliary,
    ]
    .into_iter()
    .enumerate()
    .map(|(binding, plane)| wgpu::BindGroupEntry {
        binding: binding as u32,
        resource: responses.plane(plane).expect("plane").as_entire_binding(),
    })
    .collect();
    let producer_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("GPU test split producer"),
        layout: &producer.get_bind_group_layout(0),
        entries: &entries,
    });
    let left_read = readback(&device, responses.layout().bytes(CompositionPlane::Left));
    let aux_read = readback(
        &device,
        responses.layout().bytes(CompositionPlane::Auxiliary),
    );
    let hdr_read = readback(&device, u64::from(size.y) * 256);
    for frame in [u32::MAX, 0] {
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(!responses
            .record(&pipelines, &mut encoder, &target, frame)
            .unwrap());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&producer);
            pass.set_bind_group(0, &producer_group, &[]);
            pass.dispatch_workgroups(responses.layout().pixel_count().div_ceil(64), 1, 1);
        }
        responses.publish(frame);
        assert!(responses
            .record(&pipelines, &mut encoder, &target, frame)
            .unwrap());
        assert!(!responses
            .record(&pipelines, &mut encoder, &target, frame)
            .unwrap());
        encoder.copy_buffer_to_buffer(
            responses.plane(CompositionPlane::Left).unwrap(),
            0,
            &left_read,
            0,
            responses.layout().bytes(CompositionPlane::Left),
        );
        encoder.copy_buffer_to_buffer(
            responses.plane(CompositionPlane::Auxiliary).unwrap(),
            0,
            &aux_read,
            0,
            responses.layout().bytes(CompositionPlane::Auxiliary),
        );
        encoder.copy_texture_to_buffer(
            hdr.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &hdr_read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(size.y),
                },
            },
            hdr.size(),
        );
        queue.submit([encoder.finish()]);
        let actual_left: Vec<GpuResponseOutput> = read(&device, &left_read);
        let actual_aux: Vec<GpuAuxiliaryMerge> = read(&device, &aux_read);
        let actual_hdr: Vec<u16> = read(&device, &hdr_read);
        for (i, response) in actual_left.iter().enumerate() {
            let first = 2.0 + (i % 3) as f32 * 4.0;
            for channel in response.words[..15].chunks_exact(3) {
                assert_eq!(channel, [first, 3.0, 4.0], "pixel {i}");
            }
            assert_eq!(response.words[15], 0.75);
            let offset = (i / size.x as usize) * 128 + (i % size.x as usize) * 4;
            let actual = core::array::from_fn(|channel| normal_half(actual_hdr[offset + channel]));
            let expected = if i % 11 == 0 || i % 13 == 0 {
                [1.0, 2.0, 3.0, 1.0]
            } else {
                [first * 5.0, 15.0, 20.0, 0.75]
            };
            assert_eq!(actual, expected, "HDR pixel {i}, frame {frame}");
        }
        for (i, value) in actual_aux.iter().enumerate() {
            assert_eq!(value.left[0], i as u32 | 0x80000000, "auxiliary {i}");
        }
    }
}

/// Bad targets/pipelines are rejected before wgpu validation, without consuming
/// a publication. Zero auxiliary jobs never create an empty storage binding.
#[test]
fn resident_targets_and_pipeline_identity_are_checked() {
    let Some((device, queue, render_device)) = gpu() else {
        return;
    };
    let size = UVec2::ONE;
    let pipelines = ResidentCompositionPipelines::new(&render_device);
    let other_pipelines = ResidentCompositionPipelines::new(&render_device);
    let mut responses = ViewResponseComposition::new(&render_device, &pipelines, size, 0).unwrap();
    assert!(responses.plane(CompositionPlane::Auxiliary).is_none());
    let hdr = texture(&render_device, size, wgpu::TextureFormat::Rgba16Float);
    let ids = texture(&render_device, size, wgpu::TextureFormat::Rgba32Uint);
    let wrong = texture(
        &render_device,
        UVec2::new(2, 1),
        wgpu::TextureFormat::Rgba32Uint,
    );
    assert!(matches!(
        ResponseCompositionTarget::new(&render_device, &pipelines, &responses, &hdr, &ids, &wrong),
        Err(CompositionResourceError::InvalidTarget {
            target: "visibility_metadata",
            ..
        })
    ));
    assert!(matches!(
        ResponseCompositionTarget::new(&render_device, &pipelines, &responses, &ids, &ids, &ids),
        Err(CompositionResourceError::InvalidTarget {
            target: "hdr_scene_color",
            ..
        })
    ));
    assert!(matches!(
        ResponseCompositionTarget::new(
            &render_device,
            &other_pipelines,
            &responses,
            &hdr,
            &ids,
            &ids
        ),
        Err(CompositionResourceError::PipelineMismatch)
    ));
    let target =
        ResponseCompositionTarget::new(&render_device, &pipelines, &responses, &hdr, &ids, &ids)
            .unwrap();
    let other_responses =
        ViewResponseComposition::new(&render_device, &pipelines, size, 0).unwrap();
    let other_target = ResponseCompositionTarget::new(
        &render_device,
        &pipelines,
        &other_responses,
        &hdr,
        &ids,
        &ids,
    )
    .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    responses.publish(1);
    assert_eq!(
        responses.record(&other_pipelines, &mut encoder, &target, 1),
        Err(CompositionResourceError::PipelineMismatch)
    );
    assert_eq!(
        responses.record(&pipelines, &mut encoder, &other_target, 1),
        Err(CompositionResourceError::TargetMismatch)
    );
    assert!(responses
        .record(&pipelines, &mut encoder, &target, 1)
        .unwrap());
    responses.publish(1);
    assert!(!responses
        .record(&pipelines, &mut encoder, &target, 2)
        .unwrap());
    assert!(!responses
        .record(&pipelines, &mut encoder, &target, 1)
        .unwrap());
    queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("zero auxiliary submission");
}
