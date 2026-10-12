//! Bindless buffer arrays: a `binding_array<InBuf, 4>` of storage buffers,
//! indexed at run time by a value read from another storage buffer.

use super::{class_supported, device_or_skip, output_buffer, read_u32s, require_gpu, upload_u32s};
use prism_render_driver as rhi;
use rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBinding, BufferBindingType, CommandEncoder, ComputeCommand,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, RenderDevice, RenderQueue,
    ShaderModuleDescriptor, ShaderStages,
};

/// One element buffer per array slot, each holding a single `value`.
const SHADER: &str = r"
enable wgpu_binding_array;

struct InBuf {
    value: u32,
};

@group(0) @binding(0) var<storage, read> inputs: binding_array<InBuf, 4>;
@group(0) @binding(1) var<storage, read> chosen: array<u32, 4>;
@group(0) @binding(2) var<storage, read_write> out: array<u32, 4>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    out[i] = inputs[chosen[i]].value;
}
";

#[test]
fn gpu_binding_array_selects_the_indexed_buffer() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    if !class_supported(
        &device,
        rhi::Features::BINDLESS_BUFFER_ARRAY,
        "buffer descriptor arrays",
    ) {
        return;
    }

    // Four element buffers with distinct payloads, plus a per-invocation index
    // list that permutes them: invocation 0 must read element 2, and so on.
    let payloads = [10_u32, 20, 30, 40];
    let elements: Vec<rhi::BufferId> = payloads
        .iter()
        .map(|value| upload_u32s(&device, &queue, "array-element", &[*value]))
        .collect();
    let chosen = upload_u32s(&device, &queue, "array-indices", &[2, 0, 3, 1]);
    let out = output_buffer(&device, "array-out", 4);

    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("buffer-array-layout".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::BufferArray {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(4),
                },
                count: Some(4),
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(16),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(16),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("buffer-array-group".into()),
        layout: Some(layout),
        entries: vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::BufferArray(
                    elements
                        .iter()
                        .map(|buffer| BufferBinding {
                            buffer: *buffer,
                            offset: 0,
                            size: Some(4),
                        })
                        .collect(),
                ),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer {
                    buffer: chosen,
                    offset: 0,
                    size: Some(16),
                },
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer {
                    buffer: out,
                    offset: 0,
                    size: Some(16),
                },
            },
        ],
    });

    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(SHADER));
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("buffer-array-pipeline-layout".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("buffer-array-compute".into()),
        layout: Some(pipeline_layout),
        module,
        entry_point: "main".into(),
    });

    let mut encoder = CommandEncoder::labeled("buffer-array-dispatch");
    encoder.push_compute_pass(
        Some("buffer-array".into()),
        vec![
            ComputeCommand::SetPipeline(pipeline),
            ComputeCommand::SetBindGroup {
                index: 0,
                bind_group,
                dynamic_offsets: Vec::new(),
            },
            ComputeCommand::Dispatch { x: 1, y: 1, z: 1 },
        ],
    );
    queue.submit(&[encoder.finish()]);
    assert!(device.poll_wait(), "the device stays healthy");

    // `out[i] = inputs[chosen[i]].value`, so the permutation must be applied
    // element by element: 30, 10, 40, 20.
    assert_eq!(read_u32s(&queue, out, 4), vec![30, 10, 40, 20]);

    device.destroy_bind_group(bind_group);
    device.destroy_bind_group_layout(layout);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_compute_pipeline(pipeline);
    device.destroy_shader_module(module);
    device.destroy_buffer(chosen);
    device.destroy_buffer(out);
    for buffer in elements {
        device.destroy_buffer(buffer);
    }
}
