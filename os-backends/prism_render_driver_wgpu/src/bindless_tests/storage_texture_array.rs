//! Bindless storage-texture arrays: a
//! `binding_array<texture_storage_2d<rgba8unorm, write>, 2>` written through at
//! run time.
//!
//! Each invocation paints a distinct colour into the element at its own chosen
//! index. A broken index would send both writes to the same texture, leaving the
//! other untouched, so reading both back proves the selection is per-element.

use super::{class_supported, device_or_skip, require_gpu, upload_u32s};
use prism_render_driver as rhi;
use rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, CommandEncoder, ComputeCommand, ComputePipelineDescriptor,
    Extent3d, PipelineLayoutDescriptor, RenderDevice, RenderQueue, ShaderModuleDescriptor,
    ShaderStages, StorageTextureAccess, TextureDescriptor, TextureFormat, TextureUsages,
    TextureViewDescriptor, TextureViewDimension,
};

/// Two invocations, each storing a different colour into the array element it
/// selects by reading `chosen` on the GPU.
const SHADER: &str = r"
enable wgpu_binding_array;

@group(0) @binding(0) var targets: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
@group(0) @binding(1) var<storage, read> chosen: array<u32, 2>;

@compute @workgroup_size(2)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let red = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    let green = vec4<f32>(0.0, 1.0, 0.0, 1.0);
    let colour = select(green, red, i == 0u);
    textureStore(targets[chosen[i]], vec2<i32>(0, 0), colour);
}
";

#[test]
fn gpu_binding_array_writes_the_indexed_storage_texture() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    if !class_supported(
        &device,
        rhi::Features::BINDLESS_STORAGE_TEXTURE_ARRAY,
        "storage-texture descriptor arrays",
    ) {
        return;
    }

    let storage_usage = TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC;
    let first = device.create_texture(&TextureDescriptor::new_2d(
        Extent3d::new_2d(1, 1),
        TextureFormat::Rgba8Unorm,
        storage_usage,
    ));
    let second = device.create_texture(&TextureDescriptor::new_2d(
        Extent3d::new_2d(1, 1),
        TextureFormat::Rgba8Unorm,
        storage_usage,
    ));
    let first_view = device.create_texture_view(first, &TextureViewDescriptor::default());
    let second_view = device.create_texture_view(second, &TextureViewDescriptor::default());

    // Invocation 0 paints element 0 red, invocation 1 paints element 1 green.
    let chosen = upload_u32s(&device, &queue, "storage-indices", &[0, 1]);

    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("storage-array-layout".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTextureArray {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
                },
                count: Some(2),
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: rhi::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(8),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("storage-array-group".into()),
        layout: Some(layout),
        entries: vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureViewArray(vec![first_view, second_view]),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer {
                    buffer: chosen,
                    offset: 0,
                    size: Some(8),
                },
            },
        ],
    });

    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(SHADER));
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("storage-array-pipeline-layout".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("storage-array-compute".into()),
        layout: Some(pipeline_layout),
        module,
        entry_point: "main".into(),
    });

    let mut encoder = CommandEncoder::labeled("storage-array-dispatch");
    encoder.push_compute_pass(
        Some("storage-array".into()),
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

    assert_eq!(
        queue.read_texture_rgba8(first, 1, 1),
        vec![255, 0, 0, 255],
        "element 0 received the red write"
    );
    assert_eq!(
        queue.read_texture_rgba8(second, 1, 1),
        vec![0, 255, 0, 255],
        "element 1 received the green write"
    );

    device.destroy_bind_group(bind_group);
    device.destroy_bind_group_layout(layout);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_compute_pipeline(pipeline);
    device.destroy_shader_module(module);
    device.destroy_buffer(chosen);
    device.destroy_texture_view(first_view);
    device.destroy_texture_view(second_view);
    device.destroy_texture(first);
    device.destroy_texture(second);
}
