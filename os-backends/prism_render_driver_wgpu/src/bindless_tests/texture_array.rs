//! Bindless sampled-texture arrays: a `binding_array<texture_2d<f32>, 4>`
//! indexed at run time, sampled with `textureLoad`, and read back as the chosen
//! texture's red channel.

use super::{class_supported, device_or_skip, output_buffer, read_u32s, require_gpu, upload_u32s};
use prism_render_driver as rhi;
use rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, CommandEncoder, ComputeCommand, ComputePipelineDescriptor,
    Extent3d, ImageDataLayout, PipelineLayoutDescriptor, RenderDevice, RenderQueue,
    ShaderModuleDescriptor, ShaderStages, TextureDescriptor, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension, TextureWrite,
};

/// Each array element is a 1x1 texture whose red channel is a distinct value, so
/// the readback identifies exactly which element the shader sampled.
const SHADER: &str = r"
enable wgpu_binding_array;

@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 4>;
@group(0) @binding(1) var<storage, read> chosen: array<u32, 4>;
@group(0) @binding(2) var<storage, read_write> out: array<u32, 4>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let texel = textureLoad(textures[chosen[i]], vec2<i32>(0, 0), 0);
    out[i] = u32(texel.r * 255.0 + 0.5);
}
";

/// The red channel of each array element, in slot order.
const RED_VALUES: [u8; 4] = [11, 22, 33, 44];

#[test]
fn gpu_binding_array_samples_the_indexed_texture() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    if !class_supported(
        &device,
        rhi::Features::BINDLESS_TEXTURE_ARRAY,
        "sampled-texture descriptor arrays",
    ) {
        return;
    }

    // One 1x1 element texture per array slot, uploaded through the queue.
    let mut element_textures = Vec::with_capacity(RED_VALUES.len());
    let mut element_views = Vec::with_capacity(RED_VALUES.len());
    for red in RED_VALUES {
        let texture = device.create_texture(&TextureDescriptor::new_2d(
            Extent3d::new_2d(1, 1),
            TextureFormat::Rgba8Unorm,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        ));
        let texel = [red, 0, 0, 255];
        queue.write_texture(
            TextureWrite {
                texture,
                mip_level: 0,
                origin: [0, 0, 0],
                size: Extent3d::new_2d(1, 1),
            },
            &texel,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
        );
        element_views.push(device.create_texture_view(texture, &TextureViewDescriptor::default()));
        element_textures.push(texture);
    }

    let chosen = upload_u32s(&device, &queue, "texture-indices", &[3, 1, 0, 2]);
    let out = output_buffer(&device, "texture-out", 4);

    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("texture-array-layout".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::TextureArray {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(4),
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: rhi::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(16),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: rhi::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(16),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("texture-array-group".into()),
        layout: Some(layout),
        entries: vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureViewArray(element_views.clone()),
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
        label: Some("texture-array-pipeline-layout".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("texture-array-compute".into()),
        layout: Some(pipeline_layout),
        module,
        entry_point: "main".into(),
    });

    let mut encoder = CommandEncoder::labeled("texture-array-dispatch");
    encoder.push_compute_pass(
        Some("texture-array".into()),
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

    // Invocation `i` must report the red channel of element `chosen[i]`:
    // chosen = [3, 1, 0, 2] over RED_VALUES = [11, 22, 33, 44] -> 44, 22, 11, 33.
    assert_eq!(
        read_u32s(&queue, out, 4),
        vec![44, 22, 11, 33],
        "each invocation sampled the element at its chosen index"
    );

    device.destroy_bind_group(bind_group);
    device.destroy_bind_group_layout(layout);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_compute_pipeline(pipeline);
    device.destroy_shader_module(module);
    device.destroy_buffer(chosen);
    device.destroy_buffer(out);
    for view in element_views {
        device.destroy_texture_view(view);
    }
    for texture in element_textures {
        device.destroy_texture(texture);
    }
}
