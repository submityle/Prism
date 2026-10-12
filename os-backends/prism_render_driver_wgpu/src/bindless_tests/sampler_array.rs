//! Bindless sampler arrays: a `binding_array<sampler, 2>` indexed at run time.
//!
//! The two element samplers deliberately differ only in their address mode, and
//! the shader samples far outside `[0, 1]` at a coordinate where the two modes
//! pick *different texels*. The readback therefore proves the sampler actually
//! used was the one at the chosen index, rather than a single sampler bound
//! once for all invocations.

use super::{class_supported, device_or_skip, output_buffer, read_u32s, require_gpu, upload_u32s};
use prism_render_driver as rhi;
use rhi::{
    AddressMode, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, CommandEncoder, ComputeCommand,
    ComputePipelineDescriptor, Extent3d, FilterMode, ImageDataLayout, PipelineLayoutDescriptor,
    RenderDevice, RenderQueue, SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderStages, TextureDescriptor, TextureFormat, TextureSampleType, TextureUsages,
    TextureViewDescriptor, TextureViewDimension, TextureWrite,
};

/// `chosen = [0, 1]`: invocation 0 uses the clamping sampler, invocation 1 the
/// repeating one.
const SHADER: &str = r"
enable wgpu_binding_array;

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samplers: binding_array<sampler, 2>;
@group(0) @binding(2) var<storage, read> chosen: array<u32, 2>;
@group(0) @binding(3) var<storage, read_write> out: array<u32, 2>;

@compute @workgroup_size(2)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c = textureSampleLevel(tex, samplers[chosen[i]], vec2<f32>(-0.5, 0.5), 0.0);
    out[i] = u32(c.r * 255.0 + 0.5) | (u32(c.b * 255.0 + 0.5) << 8u);
}
";

/// The 2x1 source row: texel 0 is red, texel 1 is blue, both at full intensity.
const ROW: [u8; 8] = [200, 0, 0, 255, 0, 0, 200, 255];

#[test]
fn gpu_binding_array_uses_the_indexed_sampler() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    if !class_supported(
        &device,
        rhi::Features::BINDLESS_SAMPLER_ARRAY,
        "sampler descriptor arrays",
    ) {
        return;
    }

    let texture = device.create_texture(&TextureDescriptor::new_2d(
        Extent3d::new_2d(2, 1),
        TextureFormat::Rgba8Unorm,
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    ));
    queue.write_texture(
        TextureWrite {
            texture,
            mip_level: 0,
            origin: [0, 0, 0],
            size: Extent3d::new_2d(2, 1),
        },
        &ROW,
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
    );
    let view = device.create_texture_view(texture, &TextureViewDescriptor::default());

    // Slot 0 clamps, slot 1 repeats; both select the nearest texel so the only
    // difference under test is the address mode. Sampling at `u = -0.5` is one
    // texel to the left of the row: clamping reads texel 0, repeating wraps to
    // texel 1.
    let clamping = device.create_sampler(&SamplerDescriptor {
        label: Some("clamp-nearest".into()),
        address_mode_u: AddressMode::ClampToEdge,
        ..SamplerDescriptor::default()
    });
    let repeating = device.create_sampler(&SamplerDescriptor {
        label: Some("repeat-nearest".into()),
        address_mode_u: AddressMode::Repeat,
        mag_filter: FilterMode::Nearest,
        ..SamplerDescriptor::default()
    });

    let chosen = upload_u32s(&device, &queue, "sampler-indices", &[0, 1]);
    let out = output_buffer(&device, "sampler-out", 2);

    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("sampler-array-layout".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::SamplerArray(SamplerBindingType::Filtering),
                count: Some(2),
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: rhi::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(8),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: rhi::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(8),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("sampler-array-group".into()),
        layout: Some(layout),
        entries: vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::SamplerArray(vec![clamping, repeating]),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer {
                    buffer: chosen,
                    offset: 0,
                    size: Some(8),
                },
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Buffer {
                    buffer: out,
                    offset: 0,
                    size: Some(8),
                },
            },
        ],
    });

    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(SHADER));
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("sampler-array-pipeline-layout".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("sampler-array-compute".into()),
        layout: Some(pipeline_layout),
        module,
        entry_point: "main".into(),
    });

    let mut encoder = CommandEncoder::labeled("sampler-array-dispatch");
    encoder.push_compute_pass(
        Some("sampler-array".into()),
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

    // Invocation 0 (clamping) reads the red texel: red 200, blue 0.
    // Invocation 1 (repeating) reads the blue texel: red 0, blue 200.
    let expected_clamped = 200_u32;
    let expected_repeated = 200_u32 << 8;
    assert_eq!(
        read_u32s(&queue, out, 2),
        vec![expected_clamped, expected_repeated]
    );

    device.destroy_bind_group(bind_group);
    device.destroy_bind_group_layout(layout);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_compute_pipeline(pipeline);
    device.destroy_shader_module(module);
    device.destroy_buffer(chosen);
    device.destroy_buffer(out);
    device.destroy_sampler(clamping);
    device.destroy_sampler(repeating);
    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}
