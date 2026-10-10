//! GPU-gated integration tests for the wgpu backend.
//!
//! Every test first attempts to bring up a real device via
//! [`WgpuInstance::request_default_device`]. When no adapter is present (the
//! base CI sandbox has no GPU), the test returns early and is a no-op, so the
//! suite always passes without hardware. On a machine with a GPU (e.g. a macOS
//! Metal adapter), the tests exercise the full create -> record -> submit path
//! end to end, which is the behaviour that cannot be covered by the GPU-free
//! unit tests in `convert.rs`.
//!
//! These tests depend only on `prism_render_driver_wgpu` and the frozen RHI;
//! they never name a `wgpu` type directly, relying on the no-wgpu-types
//! `request_default_device` entry point.

use prism_render_driver as rhi;
use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};
use rhi::{RenderDevice, RenderQueue};

/// Brings up a device/queue pair, or returns `None` when no GPU is available.
fn device_or_skip() -> Option<(WgpuDevice, WgpuQueue)> {
    WgpuInstance::new().request_default_device().ok()
}

#[test]
fn gpu_trivial_compute_dispatch() {
    let Some((device, queue)) = device_or_skip() else {
        return;
    };

    // A storage buffer holding four u32 values, uploaded from the CPU.
    let buffer = device.create_buffer(&rhi::BufferDescriptor {
        label: Some("compute-io".into()),
        size: 16,
        usage: rhi::BufferUsages::STORAGE
            | rhi::BufferUsages::COPY_DST
            | rhi::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let initial: [u32; 4] = [1, 2, 3, 4];
    let bytes: Vec<u8> = initial.iter().flat_map(|v| v.to_le_bytes()).collect();
    queue.write_buffer(buffer, 0, &bytes);

    // A compute shader that doubles each element in place.
    let module = device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(
        "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n\
         @compute @workgroup_size(1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
         \tdata[gid.x] = data[gid.x] * 2u;\n\
         }\n",
    ));

    let layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
        label: Some("compute-bgl".into()),
        entries: vec![rhi::BindGroupLayoutEntry {
            binding: 0,
            visibility: rhi::ShaderStages::COMPUTE,
            ty: rhi::BindingType::Buffer {
                ty: rhi::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
        label: Some("compute-pl".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });

    let pipeline = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
        label: Some("compute".into()),
        layout: Some(pipeline_layout),
        module,
        entry_point: "main".into(),
    });

    let bind_group = device.create_bind_group(&rhi::BindGroupDescriptor {
        label: Some("compute-bg".into()),
        layout: Some(layout),
        entries: vec![rhi::BindGroupEntry {
            binding: 0,
            resource: rhi::BindingResource::Buffer {
                buffer,
                offset: 0,
                size: None,
            },
        }],
    });

    let mut encoder = rhi::CommandEncoder::labeled("compute-cb");
    encoder.push_compute_pass(
        Some("double".into()),
        vec![
            rhi::ComputeCommand::SetPipeline(pipeline),
            rhi::ComputeCommand::SetBindGroup {
                index: 0,
                bind_group,
                dynamic_offsets: Vec::new(),
            },
            rhi::ComputeCommand::Dispatch { x: 4, y: 1, z: 1 },
        ],
    );
    queue.submit(&[encoder.finish()]);

    // Drive the queue to completion so any validation error surfaces here.
    device.poll_wait();

    device.destroy_compute_pipeline(pipeline);
    device.destroy_bind_group(bind_group);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_bind_group_layout(layout);
    device.destroy_shader_module(module);
    device.destroy_buffer(buffer);
}

#[test]
fn gpu_trivial_clear_render_pass() {
    let Some((device, queue)) = device_or_skip() else {
        return;
    };

    let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        rhi::Extent3d::new_2d(64, 64),
        rhi::TextureFormat::Rgba8Unorm,
        rhi::TextureUsages::RENDER_ATTACHMENT | rhi::TextureUsages::COPY_SRC,
    ));
    let view = device.create_texture_view(texture, &rhi::TextureViewDescriptor::default());

    let mut encoder = rhi::CommandEncoder::labeled("clear-cb");
    encoder.push_render_pass(
        rhi::RenderPassDescriptor {
            label: Some("clear".into()),
            color_attachments: vec![Some(rhi::ColorAttachment {
                view,
                resolve_target: None,
                load: rhi::LoadOp::Clear(rhi::Color::new(0.1, 0.2, 0.3, 1.0)),
                store: rhi::StoreOp::Store,
            })],
            depth_stencil_attachment: None,
        },
        Vec::new(),
    );
    queue.submit(&[encoder.finish()]);
    device.poll_wait();

    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}
