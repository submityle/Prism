//! M0 acceptance demo: a windowed triangle drawn through Prism's RHI.
//!
//! It brings up the wgpu backend through `prism_render_runtime`, creates a
//! surface for a `winit` window, and each frame acquires the swapchain image,
//! imports it into a [`prism_render_graph`] frame, draws a triangle, and
//! presents — the full M0 path with no Bevy and no direct `wgpu` call.
//!
//! Run with `cargo run -p render_triangle` on a machine with a display.

#![allow(
    clippy::print_stderr,
    reason = "this demo binary reports frame/resize failures to stderr and has no logging stack"
)]

extern crate alloc;

use alloc::sync::Arc;

use prism_render_driver as rhi;
use rhi::{
    ColorTargetState, FragmentState, MultisampleState, PrimitiveState, RenderDevice, RenderQueue,
    ShaderModuleDescriptor, VertexFormat, VertexState, VertexStepMode,
};
use prism_render_graph::PassFlags;
use prism_render_runtime::WgpuRuntime;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// Three `vec3<f32>` clip-space positions forming a triangle.
const TRIANGLE: [f32; 9] = [0.0, 0.6, 0.0, -0.6, -0.6, 0.0, 0.6, -0.6, 0.0];

/// Pass-through vertex stage and constant-red fragment stage.
const TRIANGLE_WGSL: &str = "\
@vertex fn vs(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {\n\
    return vec4<f32>(pos, 1.0);\n\
}\n\
@fragment fn fs() -> @location(0) vec4<f32> {\n\
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);\n\
}\n";

/// The persistent GPU objects a frame's pass draws with.
struct RenderState {
    runtime: WgpuRuntime,
    pipeline: rhi::RenderPipelineId,
    vertices: rhi::BufferId,
}

#[derive(Default)]
struct Demo {
    window: Option<Arc<Window>>,
    state: Option<RenderState>,
}

impl Demo {
    /// Brings up the RHI, surface, and pipeline for `window`.
    fn init(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut runtime =
            WgpuRuntime::wgpu(rhi::Extent3d::new_2d(width, height)).expect("wgpu runtime");
        runtime.create_surface(window.clone()).expect("create surface");
        runtime
            .configure_surface_default(width, height)
            .expect("configure surface");
        let format = runtime
            .surface_config()
            .expect("just configured")
            .format;

        let device = runtime.device();
        let queue = runtime.queue();

        let vertex_bytes: Vec<u8> = TRIANGLE.iter().flat_map(|v| v.to_le_bytes()).collect();
        let vertices = device.create_buffer(&rhi::BufferDescriptor {
            label: Some("triangle-vertices".into()),
            size: vertex_bytes.len() as u64,
            usage: rhi::BufferUsages::VERTEX | rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(vertices, 0, &vertex_bytes);

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
                targets: vec![Some(ColorTargetState::opaque(format))],
            }),
        });

        self.window = Some(window);
        self.state = Some(RenderState {
            runtime,
            pipeline,
            vertices,
        });
    }

    /// Draws and presents one frame.
    fn draw(&mut self) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let pipeline = state.pipeline;
        let vertices = state.vertices;
        let result = state.runtime.render_frame(move |graph, target| {
            graph.add_raster_pass("triangle", PassFlags::SIDE_EFFECT, move |b| {
                let _ = b.color_attachment(target);
                move |ctx| {
                    ctx.set_render_pipeline(pipeline);
                    ctx.set_vertex_buffer(0, vertices, 0);
                    ctx.draw(0..3, 0..1);
                }
            });
        });
        if let Err(error) = result {
            eprintln!("frame error: {error}");
        }
    }
}

impl ApplicationHandler for Demo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Prism RHI - triangle")
                        .with_inner_size(LogicalSize::new(800.0, 600.0)),
                )
                .expect("create window"),
        );
        self.init(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                if let Some(state) = self.state.as_mut()
                    && let Err(error) = state
                    .runtime
                    .configure_surface_default(size.width, size.height)
                {
                    eprintln!("resize failed: {error}");
                }
            }
            WindowEvent::RedrawRequested => self.draw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("create event loop");
    let mut demo = Demo::default();
    event_loop.run_app(&mut demo).expect("run event loop");
}
