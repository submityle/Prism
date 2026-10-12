//! GPU-gated tests for the RHI render spine.
//!
//! Each test first attempts to bring up a real device. When no adapter is
//! present (the usual GPU-less CI sandbox) the test returns early and is a
//! no-op, so the suite always passes without hardware. On a machine with a GPU
//! it exercises the full facade -> graph -> readback path through the frozen
//! RHI, which is exactly the ground the Bevy render stack used to provide.

use prism_render_driver as rhi;
use rhi::{
    Adapter, ColorTargetState, FragmentState, Instance, MultisampleState, PrimitiveState,
    RenderDevice, RenderQueue, ShaderModuleDescriptor, TextureState, TextureUsages, VertexFormat,
    VertexState, VertexStepMode,
};
use prism_render_driver_wgpu::WgpuInstance;
use prism_render_graph::{ImportedTexture, PassFlags, TextureDesc};

use crate::RenderRuntime;

/// The offscreen target edge length used by the readback test.
const SIZE: u32 = 64;

#[test]
fn instance_enumerates_adapters_through_the_rhi() {
    let instance = WgpuInstance::new();
    let adapters = <WgpuInstance as Instance>::enumerate_adapters(&instance);
    // A GPU-less sandbox legitimately reports nothing; skip in that case.
    for adapter in &adapters {
        let info = adapter.info();
        assert!(!info.name.is_empty(), "adapter reports a name");
        let caps = adapter.capabilities();
        assert_eq!(caps.backend, info.backend, "caps and info agree on backend");
    }
}

#[test]
fn rhi_graph_clears_imported_target() {
    let Some(rt) = RenderRuntime::wgpu(rhi::Extent3d::new_2d(SIZE, SIZE)).ok() else {
        return;
    };
    let device = rt.device();

    // An offscreen target the graph imports (so it survives culling and its
    // result outlives the frame) and clears.
    let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        rhi::Extent3d::new_2d(SIZE, SIZE),
        rhi::TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    ));
    let view = device.create_texture_view(texture, &rhi::TextureViewDescriptor::default());

    let mut graph = rt.new_graph();
    let target = graph.import_texture(
        "offscreen",
        TextureDesc::color(rhi::TextureFormat::Rgba8Unorm).with_usage(TextureUsages::COPY_SRC),
        ImportedTexture {
            texture,
            default_view: view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("clear", PassFlags::SIDE_EFFECT, |b| {
        let _ = b.color_attachment(target);
        |_ctx| {}
    });

    rt.execute(graph).expect("graph executes on the RHI");
    assert!(rt.device().poll_wait(), "the device stays healthy");

    let pixels = rt.queue().read_texture_rgba8(texture, SIZE, SIZE);
    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
    assert!(
        pixels.chunks_exact(4).all(|p| p == [0, 0, 0, 255]),
        "a freshly written version-0 target is cleared to opaque black"
    );

    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}

/// A back-to-front triangle in clip space, three `vec3<f32>` positions.
const TRIANGLE: [f32; 9] = [0.0, 0.6, 0.0, -0.6, -0.6, 0.0, 0.6, -0.6, 0.0];

/// Minimal pass-through vertex + constant-color fragment shader.
const TRIANGLE_WGSL: &str = "\
@vertex fn vs(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {\n\
    return vec4<f32>(pos, 1.0);\n\
}\n\
@fragment fn fs() -> @location(0) vec4<f32> {\n\
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);\n\
}\n";

#[test]
fn rhi_graph_draws_a_triangle() {
    let Some(rt) = RenderRuntime::wgpu(rhi::Extent3d::new_2d(SIZE, SIZE)).ok() else {
        return;
    };
    let device = rt.device();
    let queue = rt.queue();

    // Vertex buffer holding the triangle.
    let vertex_bytes: Vec<u8> = TRIANGLE.iter().flat_map(|v| v.to_le_bytes()).collect();
    let vertices = device.create_buffer(&rhi::BufferDescriptor {
        label: Some("triangle-vertices".into()),
        size: vertex_bytes.len() as u64,
        usage: rhi::BufferUsages::VERTEX | rhi::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(vertices, 0, &vertex_bytes);

    // Pipeline: pass-through vertex stage, constant-red fragment stage.
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(TRIANGLE_WGSL));
    let layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
        label: Some("triangle-pl".into()),
        bind_group_layouts: vec![],
        push_constant_ranges: vec![],
    });
    let pipeline = device.create_render_pipeline(&rhi::RenderPipelineDescriptor {
        label: Some("triangle".into()),
        layout: Some(layout),
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![rhi::VertexBufferLayout::packed(
                VertexStepMode::Vertex,
                0,
                &[VertexFormat::Float32x3],
            )],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module,
            entry_point: "fs".into(),
            targets: vec![Some(ColorTargetState::opaque(rhi::TextureFormat::Rgba8Unorm))],
        }),
    });

    // Offscreen target the graph imports, clears, and draws into.
    let target_texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        rhi::Extent3d::new_2d(SIZE, SIZE),
        rhi::TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    ));
    let target_view =
        device.create_texture_view(target_texture, &rhi::TextureViewDescriptor::default());

    let mut graph = rt.new_graph();
    let target = graph.import_texture(
        "triangle-target",
        TextureDesc::color(rhi::TextureFormat::Rgba8Unorm).with_usage(TextureUsages::COPY_SRC),
        ImportedTexture {
            texture: target_texture,
            default_view: target_view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("draw-triangle", PassFlags::SIDE_EFFECT, move |b| {
        let _ = b.color_attachment(target);
        move |ctx| {
            ctx.set_render_pipeline(pipeline);
            ctx.set_vertex_buffer(0, vertices, 0);
            ctx.draw(0..3, 0..1);
        }
    });
    rt.execute(graph).expect("triangle frame executes on the RHI");
    assert!(rt.device().poll_wait(), "the device stays healthy");

    // The compiler clears the fresh target to opaque black, then the triangle
    // paints the center red and leaves the corners black.
    let pixels = queue.read_texture_rgba8(target_texture, SIZE, SIZE);
    let at = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    };
    let center = at(SIZE / 2, SIZE / 2);
    assert!(
        center[0] > 200 && center[1] < 50 && center[2] < 50 && center[3] == 255,
        "the triangle paints the center red, got {center:?}"
    );
    let corner = at(1, 1);
    assert_eq!(
        corner,
        [0, 0, 0, 255],
        "a corner outside the triangle stays black"
    );

    device.destroy_render_pipeline(pipeline);
    device.destroy_pipeline_layout(layout);
    device.destroy_shader_module(module);
    device.destroy_buffer(vertices);
    device.destroy_texture_view(target_view);
    device.destroy_texture(target_texture);
}
