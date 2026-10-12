//! M1 GPU conformance test: the RHI's texture / view / sampler / bind-group
//! path must actually render, end to end, through a `RenderGraph` frame.
//!
//! This is the acceptance for the M1 "complete resources / views / samplers"
//! milestone, exercised the way the retired Bevy stack used to do it: a real
//! device, a real offscreen target, a real pipeline whose fragment stage samples
//! a texture, and a pixel readback that proves what the GPU produced.
//!
//! When no adapter is present (the usual GPU-less CI sandbox) the test returns
//! early and is a no-op, so the suite never requires hardware; on a machine with
//! a GPU it executes the full path and asserts against exact pixels.

use prism_render_driver as rhi;
use prism_render_driver_wgpu::WgpuInstance;
use prism_render_graph::{ImportedTexture, PassFlags, TextureDesc};
use rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, ColorTargetState, Extent3d, FragmentState, MultisampleState, PrimitiveState,
    RenderDevice, RenderPipelineDescriptor, RenderQueue, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderStages, TextureDescriptor, TextureFormat, TextureState, TextureUsages,
    TextureViewDescriptor, VertexState,
};

use prism_render_runtime::RenderRuntime;

/// Whether a missing adapter must fail the test instead of skipping it.
fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

/// The offscreen render-target edge length.
const SIZE: u32 = 64;

// The fragment shader below derives UVs by dividing `@builtin(position)` by the
// literal target size `64.0`. Keep the two in lockstep at compile time rather
// than trusting a human to update both.
const _: () = assert!(SIZE == 64);

/// Source texel colours, row-major in the uploaded 2x2 texture:
/// `(u=0,v=0)` red, `(u=1,v=0)` green, `(u=0,v=1)` blue, `(u=1,v=1)` white.
const TEXEL_00: [u8; 4] = [255, 0, 0, 255];
const TEXEL_10: [u8; 4] = [0, 255, 0, 255];
const TEXEL_01: [u8; 4] = [0, 0, 255, 255];
const TEXEL_11: [u8; 4] = [255, 255, 255, 255];

/// The tightly packed 2x2 RGBA8 source pattern: two rows of two texels.
const SOURCE_PATTERN: [u8; 16] = [
    TEXEL_00[0],
    TEXEL_00[1],
    TEXEL_00[2],
    TEXEL_00[3], // row 0, texel u=0
    TEXEL_10[0],
    TEXEL_10[1],
    TEXEL_10[2],
    TEXEL_10[3], // row 0, texel u=1
    TEXEL_01[0],
    TEXEL_01[1],
    TEXEL_01[2],
    TEXEL_01[3], // row 1, texel u=0
    TEXEL_11[0],
    TEXEL_11[1],
    TEXEL_11[2],
    TEXEL_11[3], // row 1, texel u=1
];

/// A full-screen triangle that samples `tex` through `samp`, mapping the render
/// target's pixel position into UV space.
///
/// The vertex stage synthesises a triangle large enough to cover the whole
/// `[-1, 1]^2` clip square from `@builtin(vertex_index)` alone, so the pass needs
/// no vertex buffer at all: the only resources under test are the sampler and
/// the texture. The triangle's winding makes it front-facing and the pipeline
/// additionally disables culling, so it rasterizes regardless of the backend's
/// NDC y convention.
const SHADER: &str = r"
struct VertexOutput { @builtin(position) clip: vec4<f32>, };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var result: VertexOutput;
    result.clip = vec4<f32>(positions[i], 0.0, 1.0);
    return result;
    }

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy / vec2<f32>(64.0, 64.0);
    return textureSample(tex, samp, uv);
    }
";

#[test]
fn rhi_texture_sampler_bind_group_renders_through_the_graph() {
    let Some(rt) = RenderRuntime::<WgpuInstance>::wgpu(Extent3d::new_2d(SIZE, SIZE)).ok() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    let device = rt.device();
    let queue = rt.queue();

    // 1. Upload the known 2x2 pattern through the queue upload path.
    let source = device.create_texture(&TextureDescriptor::new_2d(
        Extent3d::new_2d(2, 2),
        TextureFormat::Rgba8Unorm,
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    ));
    queue.write_texture(
        rhi::TextureWrite {
            texture: source,
            mip_level: 0,
            origin: [0, 0, 0],
            size: Extent3d::new_2d(2, 2),
        },
        &SOURCE_PATTERN,
        rhi::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(2 * 4),
            rows_per_image: Some(2),
        },
    );
    let source_view = device.create_texture_view(source, &TextureViewDescriptor::default());

    // 2. A nearest (non-comparison) sampler, so texel selection is exact. A
    //    non-comparison sampler satisfies the layout's filtering slot, and the
    //    nearest filter is what makes the exact-equality assertions below valid.
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("conformance-nearest".into()),
        ..SamplerDescriptor::default()
    });

    // 3. Bind group: one sampler + one 2D float texture, matching the shader.
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("conformance-bgl".into()),
        entries: vec![
            BindGroupLayoutEntry::sampler(0, ShaderStages::FRAGMENT),
            BindGroupLayoutEntry::texture_2d(1, ShaderStages::FRAGMENT),
        ],
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("conformance-bg".into()),
        layout: Some(layout),
        entries: vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(source_view),
            },
        ],
    });

    // 4. Pipeline: index-only vertex stage, sampling fragment stage.
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(SHADER));
    let pipeline_layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
        label: Some("conformance-pl".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: vec![],
    });
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("conformance-textured-quad".into()),
        layout: Some(pipeline_layout),
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState {
            cull_mode: None,
            ..PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module,
            entry_point: "fs".into(),
            targets: vec![Some(ColorTargetState::opaque(TextureFormat::Rgba8Unorm))],
        }),
    });

    // 5. Offscreen RGBA8 target the graph imports and draws into.
    let target_texture = device.create_texture(&TextureDescriptor::new_2d(
        Extent3d::new_2d(SIZE, SIZE),
        TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    ));
    let target_view = device.create_texture_view(target_texture, &TextureViewDescriptor::default());

    let mut graph = rt.new_graph();
    let target = graph.import_texture(
        "conformance-target",
        TextureDesc::color(TextureFormat::Rgba8Unorm).with_usage(TextureUsages::COPY_SRC),
        ImportedTexture {
            texture: target_texture,
            default_view: target_view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("sample-texture", PassFlags::SIDE_EFFECT, move |b| {
        let _ = b.color_attachment(target);
        move |ctx| {
            ctx.set_render_pipeline(pipeline);
            ctx.set_render_bind_group(0, bind_group, vec![]);
            ctx.draw(0..3, 0..1);
        }
    });

    rt.execute(graph)
        .expect("the textured frame executes on the RHI");
    assert!(rt.device().poll_wait(), "the device stays healthy");

    // 6. Read back and prove the sampled output.
    //
    //    With a nearest sampler the pixel at index `(x, y)` samples at
    //    `uv = ((x + 0.5) / 64, (y + 0.5) / 64)`, and nearest selection picks
    //    texel `floor(uv * 2)`. That is 0 while `x <= 31` and 1 for `x >= 32`
    //    (likewise `y`), with no pixel straddling the boundary and no filtering,
    //    so every output pixel is exactly one source texel and the whole image
    //    can be checked with exact byte equality.
    let pixels = queue.read_texture_rgba8(target_texture, SIZE, SIZE);
    assert_eq!(
        pixels.len(),
        (SIZE * SIZE * 4) as usize,
        "the readback covers the whole target"
    );
    let at = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    };

    // The four quadrant centres land squarely on the four source texels.
    assert_eq!(
        at(SIZE / 4, SIZE / 4),
        TEXEL_00,
        "top-left samples (u0, v0)"
    );
    assert_eq!(
        at(3 * SIZE / 4, SIZE / 4),
        TEXEL_10,
        "top-right samples (u1, v0)"
    );
    assert_eq!(
        at(SIZE / 4, 3 * SIZE / 4),
        TEXEL_01,
        "bottom-left samples (u0, v1)"
    );
    assert_eq!(
        at(3 * SIZE / 4, 3 * SIZE / 4),
        TEXEL_11,
        "bottom-right samples (u1, v1)"
    );

    // Stronger: every pixel matches its quadrant's source texel exactly.
    for y in 0..SIZE {
        for x in 0..SIZE {
            let expected = match (x >= SIZE / 2, y >= SIZE / 2) {
                (false, false) => TEXEL_00,
                (true, false) => TEXEL_10,
                (false, true) => TEXEL_01,
                (true, true) => TEXEL_11,
            };
            assert_eq!(
                at(x, y),
                expected,
                "pixel ({x}, {y}) samples its quadrant's texel"
            );
        }
    }

    // The queue is idle after the readback, so everything can be released.
    device.destroy_bind_group(bind_group);
    device.destroy_bind_group_layout(layout);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_render_pipeline(pipeline);
    device.destroy_shader_module(module);
    device.destroy_sampler(sampler);
    device.destroy_texture_view(source_view);
    device.destroy_texture(source);
    device.destroy_texture_view(target_view);
    device.destroy_texture(target_texture);
}
