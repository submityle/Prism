//! `wgpu` compute twin of the virtual-geometry triangle-gradient setup
//! ([`TriangleGradients::new`](prism_render_architecture::virtual_geometry::TriangleGradients::new)).
//!
//! The vis-buffer software rasterizer sets up per-triangle screen-space
//! gradients once and walks them incrementally across each row rather than
//! recomputing the edge functions per pixel. The CPU golden
//! [`TriangleGradients::new`](prism_render_architecture::virtual_geometry::TriangleGradients::new)
//! owns that affine setup — the signed double area, the per-column/row edge
//! increments, the normalized depth weights and the depth-plane gradients;
//! [`GpuTriangleGradients`] is the on-device twin that runs one thread per
//! triangle and returns the same fields the reference derives.
//!
//! # Culling convention
//!
//! The reference marks a triangle setup valid only when its signed double area
//! `edge(v0, v1, v2)` is strictly positive (front-facing under the y-down,
//! positive-area convention); degenerate or back-facing triangles return
//! [`None`]. The twin mirrors that: such triangles come back as [`None`] in the
//! per-triangle output vector, exactly the entries the reference culls before
//! setup.
//!
//! # Portability
//!
//! The kernel is a fixed sequence of affine subtractions, one reciprocal and
//! two three-term dot products in the portable core-`WGSL` subset, so it needs
//! no optional device feature and runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! For triangles built on integer pixel coordinates the signed edge function is
//! evaluated on exactly representable operands, so the double area and the
//! `w_x`/`w_y` increments are identical on `CPU` and `GPU` regardless of
//! fused-multiply-add contraction. When the double area is an exact power of two
//! the reciprocal is exact, each `depth * inv_area` product is exact and the
//! `z_x`/`z_y` dot sums are bit-exact, so the parity test asserts field-for-field
//! equality rather than a tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard signed-edge affine gradient setup for software
//! rasterization plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::virtual_geometry::{ScreenVertex, TriangleGradients};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one gradient-setup dispatch. Layout matches `Params`
/// in `shaders/triangle_gradients.wesl`: the triangle count then three pad
/// words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    triangle_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One triangle's three screen vertices. `36`-byte stride, matching `Triangle`
/// in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTriangle {
    x0: f32,
    y0: f32,
    d0: f32,
    x1: f32,
    y1: f32,
    d1: f32,
    x2: f32,
    y2: f32,
    d2: f32,
}

/// One triangle's gradient setup result. `56`-byte stride, matching
/// `Gradients` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuGradients {
    valid: u32,
    double_area: f32,
    wx0: f32,
    wx1: f32,
    wx2: f32,
    wy0: f32,
    wy1: f32,
    wy2: f32,
    vz0: f32,
    vz1: f32,
    vz2: f32,
    z_x: f32,
    z_y: f32,
}

/// A compiled, reusable triangle-gradient-setup pipeline.
pub struct GpuTriangleGradients {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTriangleGradients {
    /// Compiles the gradient-setup kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so it needs no
    /// optional device feature.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_triangle_gradients"),
            source: ShaderSource::Wgsl(include_str!("../shaders/triangle_gradients.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_triangle_gradients_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_triangle_gradients_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_triangle_gradients_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("setup"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTriangleGradients {
            module,
            layout,
            pipeline,
        }
    }

    /// Computes the gradient setup for each triangle in `triangles`, returning
    /// one entry per triangle in input order.
    ///
    /// Each triangle is its three [`ScreenVertex`] corners `(v0, v1, v2)`. The
    /// returned entry for triangle `t` equals
    /// [`TriangleGradients::new`](prism_render_architecture::virtual_geometry::TriangleGradients::new)`(t.0, t.1, t.2)`:
    /// [`Some`] with the same fields for a front-facing triangle, [`None`] for a
    /// degenerate or back-facing one. An empty `triangles` slice yields an empty
    /// result — storage buffers cannot be zero-sized, so it is handled by an
    /// early return.
    #[must_use]
    pub fn setup(
        &self,
        ctx: &GpuContext,
        triangles: &[(ScreenVertex, ScreenVertex, ScreenVertex)],
    ) -> Vec<Option<TriangleGradients>> {
        if triangles.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            triangle_count: u32::try_from(triangles.len())
                .expect("triangle count must fit in u32 for the GPU dispatch"),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let gpu_triangles: Vec<GpuTriangle> = triangles
            .iter()
            .map(|&(v0, v1, v2)| GpuTriangle {
                x0: v0.pos[0],
                y0: v0.pos[1],
                d0: v0.depth,
                x1: v1.pos[0],
                y1: v1.pos[1],
                d1: v1.depth,
                x2: v2.pos[0],
                y2: v2.pos[1],
                d2: v2.depth,
            })
            .collect();

        let out_bytes = (triangles.len() as u64) * (size_of::<GpuGradients>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_triangle_gradients_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let triangles_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_triangle_gradients_triangles"),
            contents: bytemuck::cast_slice(&gpu_triangles),
            usage: BufferUsages::STORAGE,
        });
        let gradients_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_triangle_gradients_gradients"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let gradients_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_triangle_gradients_gradients_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_triangle_gradients_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: triangles_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: gradients_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_triangle_gradients_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_triangle_gradients_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = params.triangle_count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&gradients_buf, 0, &gradients_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        gradients_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = gradients_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_gradients = bytemuck::cast_slice::<u8, GpuGradients>(&view).to_vec();
        drop(view);
        gradients_stage.unmap();
        debug_assert_eq!(gpu_gradients.len(), triangles.len());
        gpu_gradients
            .into_iter()
            .map(|g| {
                if g.valid == 0 {
                    None
                } else {
                    Some(TriangleGradients {
                        double_area: g.double_area,
                        w_x: [g.wx0, g.wx1, g.wx2],
                        w_y: [g.wy0, g.wy1, g.wy2],
                        vertices_z: [g.vz0, g.vz1, g.vz2],
                        z_x: g.z_x,
                        z_y: g.z_y,
                    })
                }
            })
            .collect()
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
